from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from rlhf.reward_model.core.contracts import ConfigError

_FILE_FMT = "step_{step:08d}.pt"


def rng_state() -> dict:
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state = {
        "torch_cpu" : torch.get_rng_state(),
        "python" : random.getstate(),
        "numpy" : {
            "algo" : np_algo,
            "keys" : [int(k) for k in np_keys],
            "pos" : int(np_pos),
            "has_gauss" : int(np_has_gauss),
            "cached" : float(np_cached)
        }
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state : dict) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu().to(torch.uint8))
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    np_s = state["numpy"]
    np.random.set_state((np_s["algo"], np.array(np_s["keys"], dtype = np.uint32), np_s["pos"], np_s["has_gauss"], np_s["cached"]))
    
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["torch_cuda"]])
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"].cpu().to(torch.uint8))


def _atomic_torch_save(obj, path : Path) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj : dict, path : Path) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    tmp.write_text(json.dumps(obj, indent = 2) + "\n")
    os.replace(tmp, path)


@dataclass
class CheckpointReport:
    path : str = ""
    step : int = 0
    is_best : bool = False
    pruned : list = field(default_factory = list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def save_checkpoint(
    ckpt_dir : Path, 
    model : RewardModel, 
    optimizer : torch.optim.Optimizer, 
    scheduler : Optional[torch.optim.lr_scheduler.LambdaLR] = None, 
    scaler : Optional[torch.amp.GradScaler] = None,
    *, 
    step : int, 
    epoch : int = 0,
    best_metric : Optional[float] = None, 
    is_best : bool = False,
    keep_last : Optional[int] = 2, 
    extra : Optional[dict] = None
) -> CheckpointReport:
    
    if not (isinstance(step, int) and not isinstance(step, bool) and step >= 0):
        raise ConfigError(f"step must be an int >= 0, got {step!r}")
    
    if keep_last is not None and not (isinstance(keep_last, int) and not isinstance(keep_last, bool) and keep_last >= 1):
        raise ConfigError(f"keep_last must be an int >= 1, got {keep_last!r}")
    
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents = True, exist_ok = True)

    payload = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict() if scheduler is not None else None,
        "scaler" : scaler.state_dict() if scaler is not None else None,
        "rng" : rng_state(),
        "step" : step,
        "epoch" : int(epoch),
        "best_metric" : None if best_metric is None else float(best_metric),
        "extra" : dict(extra) if extra else {}
    }

    path = ckpt_dir / _FILE_FMT.format(step=step)
    
    _atomic_torch_save(payload, path)
    _atomic_json({"file" : path.name, "step" : step}, ckpt_dir / "latest.json")
    
    if is_best:
        _atomic_json({"file" : path.name, "step" : step}, ckpt_dir / "best.json")
    
    pruned = []

    if keep_last is not None:
        protected = {path.name}
        best_ptr = ckpt_dir / "best.json"
        if best_ptr.exists():
            protected.add(json.loads(best_ptr.read_text())["file"])
        files = sorted(ckpt_dir.glob("step_*.pt"))
        
        for old in files[: -keep_last] if len(files) > keep_last else []:
            if old.name not in protected:
                old.unlink()
                pruned.append(old.name)
    
    return CheckpointReport(path = str(path), step = step, is_best = is_best, pruned = pruned)


def load_checkpoint(ckpt_dir_or_file, which : str = "latest") -> dict:
    p = Path(ckpt_dir_or_file)

    if p.is_file():
        target = p
    else:
        if which not in ("latest", "best"):
            raise ConfigError(f"which must be latest/best, got {which!r}")
        ptr = p / f"{which}.json"
        
        if not ptr.exists():
            raise ConfigError(f"no {which}.json in {p}. nothing to resume from. pass a checkpoint file path directly")
        target = p / json.loads(ptr.read_text())["file"]
        if not target.exists():
            raise ConfigError(f"{which}.json points at missing file {target.name}")
    return torch.load(target, map_location = "cpu", weights_only = True)


def restore(
    state : dict, 
    model : RewardModel, 
    optimizer : Optional[torch.optim.Optimizer] = None, 
    scheduler : Optional[torch.optim.lr_scheduler.LambdaLR] = None, 
    scaler : Optional[torch.amp.GradScaler] = None,
    with_rng : bool = True
):
    model.load_state_dict(state["model"])
    
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ConfigError("asked to restore a scheduler but the checkpoint saved none")
        scheduler.load_state_dict(state["scheduler"])

    if scaler is not None:
        if state["scaler"] is None:
            raise ConfigError("asked to restore a scaler but the checkpoint saved none")
        scaler.load_state_dict(state["scaler"])

    if with_rng:
        set_rng_state(state["rng"])
    
    return state["step"], state["epoch"], state["best_metric"], state["extra"]
