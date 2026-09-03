from __future__ import annotations
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Union
import torch
from torch import nn

from rlhf.reward_model.core.contracts import ConfigError
from rlhf.reward_model.core.device import ExecutionPlan, seed_everything
from rlhf.reward_model.core.logging import RunLogger
from rlhf.reward_model.model.losses import bt_loss
from rlhf.reward_model.training.checkpoint import load_checkpoint, restore, save_checkpoint
from rlhf.reward_model.training.evaluate import evaluate
from rlhf.reward_model.training.optim import build_optimizer, build_scheduler


@dataclass
class TrainConfig:
    epochs : int = 1
    batch_size : int = 8
    accm_steps : int = 1
    eval_batch_size : int = 16
    lr : float = 1e-4
    weight_decay : float = 0.01
    sched : Literal["linear", "constant", "cosine"] = "cosine"
    warmup_steps : int = 100
    clip_norm : float = 1.0
    log_every : int = 10
    eval_every : int = 50
    early_stop_patience : int = 5
    min_delta : float = 0.0
    keep_last : Optional[int] = None
    disable_dropout : bool = True

    def to_dict(self):
        return dict(self.__dict__)


def _check_range(
    name : str,
    val,
    dtype : Literal[int, float], 
    min_val : Optional[Union[int, float]] = None, 
    max_val : Optional[Union[int, float]] = None
):  
    if isinstance(val, int) and isinstance(val, bool):
        raise ConfigError(f"{name} : {val} expected to be of type {dtype} but instead is {type(val)}")
        
    if not isinstance(val, dtype):
        raise ConfigError(f"{name} : {val} expected to be of type {dtype} but instead is {type(val)}")
    
    if min_val is not None and val < min_val:
        raise ConfigError(f"{name} : {val} must be >= than {min_val}")

    if max_val is not None and val > max_val:
        raise ConfigError(f"{name} : {val} must be<= than {max_val}")


def disable_dropout(model : nn.Module):
    num = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0
            num += 1
    return num

def epoch_permutation(n : int, seed : int, epoch : int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator = g)


@dataclass
class TrainReport:
    steps : int = 0
    epochs_run : int = 0
    best_metric : Optional[float] = None
    best_step : Optional[int] = None
    final_accuracy : float = 0.0
    early_stopped : bool = False
    wall_seconds : float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)




class Trainer:
    def __init__(
        self,
        cfg : TrainConfig,
        model : RewardModel,
        train_store : PretokenizedPairs,
        val_store : PretokenizedPairs,
        plan : ExecutionPlan,
        logger : RunLogger,
        out_dir : Union[str, Path]
    ):
        _check_range("accm_steps", cfg.accm_steps, int, min_val = 1)
        _check_range("batch_size", cfg.batch_size, int, min_val = 1)
        _check_range("epochs", cfg.epochs, int, min_val = 1)
        _check_range("eval_batch_size", cfg.eval_batch_size, int, min_val = 1)

        _check_range("log_every", cfg.log_every, int, min_val = 1)
        _check_range("eval_every", cfg.eval_every, int, min_val = 0)
        _check_range("early_stop_patience", cfg.early_stop_patience, int, min_val = 0)
        if cfg.keep_last is not None:
           _check_range("keep_last", cfg.keep_last, int, 1)
        
        if not (isinstance(cfg.min_delta, (int, float)) and cfg.min_delta >= 0):
            raise ConfigError(f"min_delta must be >= 0, got {cfg.min_delta!r}")

        if not (isinstance(cfg.clip_norm, (int, float)) and cfg.clip_norm >= 0):
            raise ConfigError(f"clip_norm must be >= 0 (0 disables), got {cfg.clip_norm!r}")

        if len(train_store) == 0:
            raise ConfigError("train_store is empty")
        
        if len(val_store) == 0:
            raise ConfigError("val_store is empty")
        
        self.cfg = cfg
        self.plan = plan
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.train_store = train_store
        self.val_store = val_store

        self.model = model.to(plan.device)
        self.dropout_zero = disable_dropout(self.model) if cfg.disable_dropout else 0
        self.batches_per_epoch = math.ceil(len(train_store) / cfg.batch_size)
        self.steps_per_epochs = math.ceil(self.batches_per_epoch / cfg.accm_steps)
        self.total_steps = self.steps_per_epochs * cfg.epochs

        self.optimizer, self.optim_report = build_optimizer(self.model, lr = cfg.lr, weight_decay = cfg.weight_decay)
        total_steps = self.total_steps if cfg.sched in ("cosine", "linear") else None
        self.sched, self.sched_report = build_scheduler(self.optimizer, cfg.sched, cfg.warmup_steps, total_steps)
        self.scaler = torch.amp.grad_scaler.GradScaler(device = plan.torch_device().type) if plan.amp.grad_scaler else None
        self._amp = lambda : (torch.autocast(device_type = plan.torch_device().type, dtype = plan.torch_dtype())) if plan.amp.autocast else nullcontext()
    

    def _train_groups(self, epochs : int, start_idx : int = 0):
        perm = epoch_permutation(len(self.train_store), self.plan.seed, epochs)
        bs = self.cfg.batch_size
        chunks = [perm[i : i + bs] for i in range(0, len(perm), bs)][start_idx :]
        k = self.cfg.accm_steps
        for i in range(0, len(chunks), k):
            yield start_idx + i, chunks[i : i + k]
        
    def _val_batches(self):
        bs = self.cfg.eval_batch_size
        for i in range(0, len(self.val_store), bs):
            yield self.val_store.batch(list(range(i, min(i + bs, len(self.val_store)))))
    
    def _run_eval(self, step : int, epoch : int, train_start_idx : int):
        res = evaluate(
            self.model, 
            self._val_batches(), 
            device = self.plan.device, 
            autocast_dtype = self.plan.torch_dtype() if self.plan.amp.autocast else None
        )

        self.logger.log(step, **res.flat("eval"))
        acc = res.overall.accuracy

        improved = self._best is None or acc > self._best + self.cfg.min_delta
        if improved:
            self._best, self._best_step, self._stale = acc, step, 0
        else:
            self._stale += 1
        
        save_checkpoint(
            ckpt_dir = self.ckpt_dir,
            model = self.model,
            optimizer = self.optimizer,
            scheduler = self.sched,
            scaler = self.scaler,
            step = step, 
            epoch = epoch,
            best_metric = self._best, 
            is_best = improved,
            keep_last = self.cfg.keep_last,
            extra = {"epoch": epoch, "train_start_idx": train_start_idx}
        )
        return res, improved
    
    def fit(self, resume = False) -> TrainReport:
        t0 = time.perf_counter()
        self._best, self._best_step, self._stale = None, None, 0
        step, start_epoch, start_idx = 0, 0, 0

        if resume:
            state = load_checkpoint(self.ckpt_dir, which = "latest")
            step, _, self._best, extra = restore(state, self.model, self.optimizer, self.sched, self.scaler)

            for pstate in self.optimizer.state.values():
                for k, v in pstate.items():
                    if torch.is_tensor(v):
                        pstate[k] = v.to(self.plan.device)
            
            start_epoch, start_idx = extra["epoch"], extra["train_start_idx"]
            self.logger.say(f"resumed from step {step} (epoch {start_epoch}, cursor {start_idx})")
        else:
            seed_everything(self.plan.seed, self.plan.deterministic)
        

        self.logger.stamp("train_config", self.cfg.to_dict())
        self.logger.stamp("plan", self.plan.to_dict())
        self.logger.stamp("optim", self.optim_report.to_dict())
        self.logger.stamp("sched", self.sched_report.to_dict())
        if hasattr(self.model, "report"):
            self.logger.stamp("model", self.model.report.to_dict())
        self.logger.stamp("trainer", {"batches_per_epoch" : self.batches_per_epoch,
                                      "steps_per_epoch" : self.steps_per_epochs,
                                      "total_steps" : self.total_steps,
                                      "dropout_zeroed" : self.dropout_zero})

        self.model.train()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        run_loss, run_pairs, n_since_log, t_log = 0.0, 0, 0, time.perf_counter()
        stopped = False
        epochs_run = start_epoch

        for ep in range(start_epoch, self.cfg.epochs):
            cur_0 = start_idx if ep == start_epoch else 0
            for cur, grp in self._train_groups(ep, cur_0):
                n_group = int(sum(len(c) for c in grp))
                self.optimizer.zero_grad(set_to_none = True)
                grp_loss = 0.0

                for chunk in grp:
                    batch = self.train_store.batch(chunk.tolist()).to(self.plan.device)
                    with self._amp():
                        rc, rr = self.model.forward_pair(batch)
                        loss = bt_loss(rc, rr, reduction = "sum") / n_group
                
                    (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
                    grp_loss += float(loss.detach())

                if self.cfg.clip_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg.clip_norm)
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.sched.step()

                step += 1
                run_loss += grp_loss
                run_pairs += n_group
                n_since_log += 1

                if step % self.cfg.log_every == 0:
                    dt = time.perf_counter() - t_log
                    self.logger.log(
                        step, 
                        loss = run_loss / n_since_log,
                        lr = self.optimizer.param_groups[0]["lr"],
                        epoch = ep,
                        pairs_per_sec = run_pairs / dt if dt > 0 else 0.0
                    )
                    
                    run_loss, run_pairs, n_since_log, t_log = 0.0, 0, 0, time.perf_counter()
                
                if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                    _, _ = self._run_eval(step, ep, cur + len(grp))
                    self.model.train()

                    if self.cfg.early_stop_patience > 0 and self._stale >= self.cfg.early_stop_patience:
                        self.logger.say(f"early stop at step {step} ({self._stale} evals without improvement)")
                        stopped = True
                        break
                
            epochs_run = ep + 1
            if stopped:
                break
        
        final, _ = self._run_eval(step, epochs_run, 0)
        report = TrainReport(
            steps = step, 
            epochs_run = epochs_run,
            best_metric = self._best, 
            best_step = self._best_step,
            final_accuracy = final.overall.accuracy,
            early_stopped = stopped,
            wall_seconds = time.perf_counter() - t0
        )
        self.logger.stamp("train_report", report.to_dict())
        return report
