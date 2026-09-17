from __future__ import annotations
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, Optional, TYPE_CHECKING
import torch

from rlhf.core.contracts import ConfigError
from rlhf.core.device import ExecutionPlan, amp_context, seed_everything
from rlhf.core.logging import RunLogger
from rlhf.core.policy.lm import disable_dropout, enable_gradient_checkpointing, sequence_logprobs
from rlhf.core.preference.collate import dataset_digest
from rlhf.core.scoring.rewards import combine_rewards, resolve_reward_weights
from rlhf.core.training.checkpoint import check_resume_state, load_checkpoint, restore, resume_source, save_checkpoint
from rlhf.core.training.config import _check_range
from rlhf.core.training.optim import bf16_update_report, build_optimizer, build_scheduler
from rlhf.dpo.evaluate import EvalReport, evaluate
from rlhf.dpo.losses import dpo_loss
from rlhf.dpo.reference import CachedReference, assert_step_zero

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from rlhf.core.policy.lm import PolicyReport
    from rlhf.core.preference.schema import PairView
    from rlhf.core.scoring.rewards import CustomRewardFunction
    from rlhf.dpo.collate import DPOCollator
    from rlhf.dpo.config import DPOConfig
    from rlhf.dpo.reference import Reference


PathLike = Union[str, Path]

TRAIN_KEYS = (
    "accuracy", 
    "delta_mean", 
    "reward_margin", 
    "grad_weight_mean", 
    "frac_pairs_saturated",
    "policy_logp_chosen", 
    "policy_logp_rejected", 
    "logratio_chosen", 
    "logratio_rejected", 
    "sft_term"
)


def epoch_permutation(n : int, seed : int, epoch : int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator = g)


def flatten_eval(rep : EvalReport, prefix : str = "eval") -> dict:
    out = {f"{prefix}/{k}" : v for k, v in rep.preference.to_dict().items()}
    out[f"{prefix}/seconds"] = rep.seconds
    
    for label, m in rep.slices.items():
        for k in ("n_pairs", "accuracy", "delta_mean", "frac_winner_below_ref"):
            out[f"{prefix}/slice/{label}/{k}"] = getattr(m, k)
    
    g = rep.generation
    if g is not None:
        out[f"{prefix}/gen/n_samples"] = g.n_samples
        out[f"{prefix}/gen/mean_new_tokens"] = g.mean_new_tokens
        out[f"{prefix}/gen/frac_empty"] = g.frac_empty
        out[f"{prefix}/gen/reward_mean"] = g.reward_mean
        out[f"{prefix}/gen/reward_min"] = g.reward_min
        out[f"{prefix}/gen/reward_max"] = g.reward_max
        if g.reward is not None:
            for nm, mu in zip(g.reward.names, g.reward.means):
                out[f"{prefix}/gen/reward/{nm}"] = mu
    return out


@dataclass
class DPOTrainReport:
    steps : int = 0
    epochs_run : int = 0
    best_metric : Optional[float] = None
    best_step : Optional[int] = None
    final_accuracy : float = 0.0
    final_frac_winner_below_ref : float = 0.0
    step_zero_max_dev : Optional[float] = None
    early_stopped : bool = False
    wall_seconds : float = 0.0

    def to_dict(self):
        return dict(self.__dict__)


class DPOTrainer:
    def __init__(
        self,
        cfg : DPOConfig,
        policy_model : PreTrainedModel,
        tokenizer : PreTrainedTokenizerBase,
        collator : DPOCollator,
        train_pairs : Sequence[PairView],
        val_pairs : Sequence[PairView],
        train_reference : Reference,
        val_reference : Reference,
        plan : ExecutionPlan,
        out_dir : PathLike,
        logger : Optional[RunLogger] = None,
        reward_functions : Optional[Sequence[CustomRewardFunction]] = None,
        gen_prompts : Optional[Sequence[str]] = None,
        policy_report : Optional[PolicyReport] = None,
        step_zero_tol : float = 1e-4
    ):
        train_cfg = cfg.train

        _check_range("accm_steps", train_cfg.accm_steps, int, min_val = 1)
        _check_range("batch_size", train_cfg.batch_size, int, min_val = 1)
        _check_range("epochs", train_cfg.epochs, int, min_val = 1)
        _check_range("eval_batch_size", train_cfg.eval_batch_size, int, min_val = 1)
        _check_range("log_every", train_cfg.log_every, int, min_val = 1)
        _check_range("eval_every", train_cfg.eval_every, int, min_val = 0)
        _check_range("early_stop_patience", train_cfg.early_stop_patience, int, min_val = 0)
        
        if train_cfg.keep_last is not None:
            _check_range("keep_last", train_cfg.keep_last, int, min_val = 1)

        check_resume_state(train_cfg.resume_state)

        if not (isinstance(train_cfg.min_delta, (int, float)) and train_cfg.min_delta >= 0):
            raise ConfigError(f"min_delta must be >= 0, got {train_cfg.min_delta!r}")

        if not (isinstance(train_cfg.clip_norm, (int, float)) and train_cfg.clip_norm >= 0):
            raise ConfigError(f"clip_norm must be >= 0 (0 disables), got {train_cfg.clip_norm!r}")

        if not (isinstance(step_zero_tol, (int, float)) and step_zero_tol >= 0):
            raise ConfigError(f"step_zero_tol must be >= 0, got {step_zero_tol!r}")

        if len(train_pairs) == 0:
            raise ConfigError("train_pairs is empty")
        if len(val_pairs) == 0:
            raise ConfigError("val_pairs is empty")
        

        for name, ref, pairs in [
            ("train", train_reference, train_pairs),
            ("val", val_reference, val_pairs)
        ]:
            if isinstance(ref, CachedReference):
                want_digest = dataset_digest(pairs)
                if want_digest != ref.report.data_digest:
                    raise ConfigError(
                        f"{name} was built over a different dataset than the pairs it will serve.\n"
                        f"    cache : {ref.report.data_digest}\n"
                        f"    pairs : {want_digest}\n"
                        f"pair uids are positional, so a cache for the wrong split answers every "
                        f"lookup with another pair's anchor and never raises. if the train cache "
                        f"was passed as val_reference, build a second cache over the val pairs"
                    )
        

        self.cfg = cfg
        self.plan = plan
        self.out_dir = Path(out_dir)
        self.logger = logger if logger is not None else RunLogger(out_dir = out_dir, run_name = self.cfg.run_name)
        self.ckpt_dir = self.logger.dir / "checkpoints"
        self.tokenizer = tokenizer
        self.collator = collator
        self.train_pairs = list(train_pairs)
        self.val_pairs = list(val_pairs)
        self.train_reference = train_reference
        self.val_reference = val_reference
        self.gen_prompts = list(gen_prompts or [])
        self.policy_report = policy_report
        self.step_zero_tol = float(step_zero_tol)

        reward_fns, _ = resolve_reward_weights(
            reward_functions, 
            self.cfg.reward.reward_wts or None, 
            self.cfg.reward.normalize_reward_func_wts
        )
        self.reward_functions = reward_fns

        self.model = policy_model.to(plan.device)
        for ref in (self.train_reference, self.val_reference):
            live_model = getattr(ref, "model", None)
            if live_model is not None:
                live_model.to(plan.device)
        
        self.dropout_zero = disable_dropout(self.model) if train_cfg.disable_dropout else 0
        self.gradient_checkpointing = enable_gradient_checkpointing(self.model) if self.cfg.policy.gradient_checkpointing else False
        if self.gradient_checkpointing:
            self.logger.say("gradient checkpointing on: activations are dropped in the forward and recomputed in the backward so each step pays one extra forward for a smaller peak memory")
        self.batches_per_epoch = math.ceil(len(train_pairs) / self.cfg.train.batch_size)
        self.steps_per_epoch = math.ceil(self.batches_per_epoch / self.cfg.train.accm_steps)
        self.total_steps = self.steps_per_epoch * self.cfg.train.epochs

        self.optimizer, self.optimizer_report = build_optimizer(self.model, self.cfg.train.lr, self.cfg.train.weight_decay)
        self.sched, self.sched_report = build_scheduler(self.optimizer, self.cfg.train.sched, self.cfg.train.warmup_steps, self.total_steps)

        if plan.weights_dtype == "bfloat16":
            stuck, n_trainable = bf16_update_report(self.model, self.cfg.train.lr)
            self.logger.say(f"bf16 weights: {stuck:.1%} of {n_trainable} trainable parameters sit above 512 x lr so a step of lr is under half a bf16 ulp and rounds to nothing")

        self.scaler = torch.amp.grad_scaler.GradScaler(device = plan.torch_device().type) if plan.amp.grad_scaler else None
        self._amp = lambda : amp_context(plan.torch_device(), plan.torch_autocast_dtype())

        eval_batch_size = self.cfg.train.eval_batch_size
        self.val_batches = [collator(self.val_pairs[i : i + eval_batch_size]) for i in range(0, len(self.val_pairs), eval_batch_size)]
    
    def _train_groups(self, epoch : int, start_idx : int = 0):
        train_idxs = epoch_permutation(len(self.train_pairs), self.plan.seed, epoch)
        bs = self.cfg.train.batch_size
        chunks_idxs = [train_idxs[i : i + bs] for i in range(0, len(train_idxs), bs)][start_idx :]
        accm_steps = self.cfg.train.accm_steps
        for i in range(0, len(chunks_idxs), accm_steps):
            yield start_idx + i, chunks_idxs[i : i + accm_steps]
    
    def _collate(self, idxs : torch.Tensor):
        return self.collator([self.train_pairs[i] for i in idxs.tolist()]).to(self.plan.device)
    
    def _step_zero_identity(self):
        ref_ckpt = self.cfg.reference.reference_ckpt

        if ref_ckpt and ref_ckpt != self.cfg.policy.model_ckpt:
            self.logger.say("step-0 identity check skipped: reference_ckpt differs from the policy init, log-ratios need not be zero")
            return None
        
        batch = self._collate(torch.arange(min(self.cfg.train.batch_size, len(self.train_pairs))))
        worst_chosen, worst_rejected = assert_step_zero(self.train_reference, self.model, batch, self.cfg.loss.length_norm, self.step_zero_tol, autocast_dtype = self.plan.torch_autocast_dtype())
        dev = float(max(float(worst_chosen), float(worst_rejected)))
        self.logger.say(f"step-0 identity holds: max |log pi - log pi_ref| = {dev:.3e}   (tol {self.step_zero_tol:g}, {self.train_reference.report.backend} reference)")
        return dev
    
    def _reward_smoke_test(self) -> None:
        if not self.reward_functions:
            return
        rew = self.cfg.reward
        sample = self.val_pairs[: 2]
        prompts, responses = [p.prompt for p in sample], [p.chosen for p in sample]
        _, rep = combine_rewards(prompts, responses, self.reward_functions, rew.reward_wts or None, rew.normalize_reward_func_wts)
        combine_rewards([], [], self.reward_functions, rew.reward_wts or None, rew.normalize_reward_func_wts)
        self.logger.say(f"reward smoke test passed: {list(rep.names)} on {len(sample)} samples and an empty batch")
    
    def _run_eval(self, step : int, epoch : int, train_start_idx : int, select : bool = True):
        eval_rep = evaluate(
            self.cfg,
            self.model,
            self.val_reference,
            self.val_batches,
            self.tokenizer if self.gen_prompts else None,
            self.gen_prompts or None,
            self.reward_functions or None,
            self.plan.device,
            with_slices = True,
            seed = self.plan.seed,
            autocast_dtype = self.plan.torch_autocast_dtype()
        )

        self.logger.log(step, **flatten_eval(eval_rep))
        self.logger.stamp("eval_latest", {"step" : step, **eval_rep.to_dict()})
        if not select:
            return eval_rep, False

        eval_acc = eval_rep.preference.accuracy
        
        improved = self._best is None or eval_acc > self._best + self.cfg.train.min_delta
        if improved:
            self._best, self._best_step, self._stale = eval_acc, step, 0
            self.logger.stamp("eval_best", {"step" : step, **eval_rep.to_dict()})
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
            keep_last = self.cfg.train.keep_last,
            extra = {"epoch" : epoch, "train_start_idx" : train_start_idx, "best_step" : self._best_step, "stale" : self._stale},
            resume_state = self.cfg.train.resume_state
        )
        return eval_rep, improved
    
    def fit(self, resume : bool = False) -> DPOTrainReport:
        t_start = time.perf_counter()
        self._best, self._best_step, self._stale = None, None, 0
        step, start_epoch, start_idx = 0, 0, 0
        train_cfg = self.cfg.train

        if resume:
            state = load_checkpoint(self.ckpt_dir, which = resume_source(train_cfg.resume_state))
            step, _, best_metrics, extra = restore(state, self.model, self.optimizer, self.sched, self.scaler)
            self._best = best_metrics
            self._best_step = extra.get("best_step", None)
            self._stale = extra.get("stale", 0)

            for pstate in self.optimizer.state.values():
                for k, v in pstate.items():
                    if torch.is_tensor(v):
                        pstate[k] = v.to(self.plan.device)
            
            start_epoch, start_idx = extra["epoch"], extra["train_start_idx"]
            self.logger.say(f"resumed from step {step} (epoch {start_epoch}, cursor {start_idx})")
        else:
            seed_everything(self.plan.seed, self.plan.deterministic)
        
        self.logger.stamp("train_config", train_cfg.to_dict())
        self.logger.stamp("plan", self.plan.to_dict())
        self.logger.stamp("optim", self.optimizer_report.to_dict())
        self.logger.stamp("sched", self.sched_report.to_dict())
        self.logger.stamp("collate", self.collator.report.to_dict())
        self.logger.stamp("reference", {"train" : self.train_reference.report.to_dict(), "val" : self.val_reference.report.to_dict()})

        if self.policy_report is not None:
            self.logger.stamp("policy", self.policy_report.to_dict())
        self.logger.stamp("trainer", {
            "batches_per_epoch" : self.batches_per_epoch,
            "steps_per_epoch" : self.steps_per_epoch,
            "total_steps" : self.total_steps,
            "dropout_zeroed" : self.dropout_zero,
            "gradient_checkpointing" : self.gradient_checkpointing,
            "n_train_pairs" : len(self.train_pairs),
            "n_val_pairs" : len(self.val_pairs),
            "n_gen_prompts" : len(self.gen_prompts),
            "reward_functions" : [getattr(f, "name", type(f).__name__) for f in self.reward_functions],
            "step_zero_tol" : self.step_zero_tol,
        })

        step_zero_dev = None
        if not resume:
            step_zero_dev = self._step_zero_identity()
            self._reward_smoke_test()

            if train_cfg.eval_every > 0:
                self._run_eval(0, 0, 0, select = False)
            
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        run_loss, run_pairs, n_since_log, t_log = 0.0, 0, 0, time.perf_counter()
        acc = {k : 0.0 for k in TRAIN_KEYS}

        acc_pairs = 0
        stopped = False
        epochs_run = start_epoch
        last_eval_step, last_eval_rep = -1, None

        for ep in range(start_epoch, train_cfg.epochs):
            start = start_idx if ep == start_epoch else 0
            for st, grp in self._train_groups(ep, start):
                n_grp = int(sum(len(c) for c in grp))
                self.optimizer.zero_grad(set_to_none = True)
                grp_loss = 0.0

                for chunk in grp:
                    batch = self._collate(chunk)
                    with self._amp():
                        logits = self.model(input_ids = batch.input_ids, attention_mask = batch.attention_mask).logits
                
                    logps = sequence_logprobs(
                        self.model, 
                        batch.input_ids, 
                        batch.attention_mask, 
                        batch.completion_mask, 
                        self.cfg.loss.length_norm, 
                        logits.float()
                    )

                    pc, pr = logps[: batch.B], logps[batch.B :]

                    with torch.no_grad():
                        rc, rr = self.train_reference(batch)
                    
                    loss, loss_metrics = dpo_loss(pc, pr, rc, rr, self.cfg.loss.beta, self.cfg.loss.sft_wt, reduction = "mean")
                    loss = loss * (batch.B / n_grp)
                    (self.scaler.scale(loss) if self.scaler is not None else loss).backward()

                    grp_loss += float(loss.detach())
                    for k in TRAIN_KEYS:
                        acc[k] += getattr(loss_metrics, k) * batch.B
                    acc_pairs += batch.B
                
                if train_cfg.clip_norm > 0.0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.clip_norm)
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.sched.step()

                step += 1
                run_loss += float(grp_loss)
                run_pairs += n_grp
                n_since_log += 1
            
                if step % train_cfg.log_every == 0:
                    dt = time.perf_counter() - t_log
                    self.logger.log(
                        step,
                        loss = run_loss / n_since_log,
                        lr = self.optimizer.param_groups[0]["lr"],
                        epoch = ep,
                        pairs_per_sec = run_pairs / dt if dt > 0 else 0.0,
                        **{f"train/{k}" : acc[k] / max(acc_pairs, 1) for k in TRAIN_KEYS},
                    )
                    run_loss, run_pairs, n_since_log, t_log = 0.0, 0, 0, time.perf_counter()
                    acc = {k : 0.0 for k in TRAIN_KEYS}
                    acc_pairs = 0
                
                if train_cfg.eval_every > 0 and step % train_cfg.eval_every == 0:
                    last_eval_rep, _ = self._run_eval(step, ep, st + len(grp))
                    last_eval_step = step
                    self.model.train()
                    if train_cfg.early_stop_patience > 0 and self._stale >= train_cfg.early_stop_patience:
                        self.logger.say(f"early stop at step {step} ({self._stale} evals without improvement)")
                        stopped = True
                        break



            epochs_run = ep + 1
            if stopped:
                break
        
        final = last_eval_rep if last_eval_step == step else self._run_eval(step, epochs_run, 0)[0]
        report = DPOTrainReport(
            steps = step,
            epochs_run = epochs_run,
            best_metric = self._best,
            best_step = self._best_step,
            final_accuracy = final.preference.accuracy,
            final_frac_winner_below_ref = final.preference.frac_winner_below_ref,
            step_zero_max_dev = step_zero_dev,
            early_stopped = stopped,
            wall_seconds = time.perf_counter() - t_start,
        )
        self.logger.stamp("train_report", report.to_dict())
        return report


def render(rep : DPOTrainReport, width : int = 76) -> str:
    bar = "=" * width
    best = "n/a" if rep.best_metric is None else f"{rep.best_metric:.4f} @ step {rep.best_step}"
    dev = "skipped" if rep.step_zero_max_dev is None else f"{rep.step_zero_max_dev:.3e}"
    L = [bar, "DPO TRAIN", bar,
         f"  steps         : {rep.steps:,}   epochs {rep.epochs_run}   {rep.wall_seconds:.1f}s"
         + ("   early stopped" if rep.early_stopped else ""),
         f"  best accuracy : {best}",
         f"  final         : accuracy {rep.final_accuracy:.4f}   winner below ref {rep.final_frac_winner_below_ref:.1%}",
         f"  step-0 dev    : {dev}",
         bar]
    if rep.final_frac_winner_below_ref > 0.5:
        L.insert(-1, "  !  the winner's likelihood fell on most pairs. accuracy is not the story here.")
    return "\n".join(L)
