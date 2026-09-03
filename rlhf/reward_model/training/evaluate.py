from __future__ import annotations
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Iterable, Optional
import torch

from rlhf.reward_model.core.contracts import ConfigError
from rlhf.reward_model.training.metrics import PairMetrics, pair_metrics


def slice_metrics(r_chosen : torch.Tensor, r_rejected : torch.Tensor, slices : dict):
    if len(slices) != r_chosen.shape[0]:
        raise ConfigError(f"len(slices) must be equal to len(rewards) but slices has {len(slices)} entries for {r_chosen.shape[0]} pairs")
    
    groups : dict = {}

    for i, dct in enumerate(slices):
        for k, v in dct.items():
            if v == "":
                continue
            groups.setdefault(f"{k}={v}", []).append(i)
    
    out = {}
    for lbl in sorted(groups):
        idx = torch.tensor(groups[lbl], dtype = torch.long)
        out[lbl] = pair_metrics(r_chosen[idx], r_rejected[idx])
    return out


@dataclass
class EvalResult:
    overall : PairMetrics
    slices : dict = field(default_factory=dict)
    n_pairs : int = 0
    n_batches : int = 0
    seconds : float = 0.0

    def to_dict(self) -> dict:
        return {
            "overall" : self.overall.to_dict(),
            "slices" : {k : v.to_dict() for k, v in self.slices.items()},
            "n_pairs" : self.n_pairs,
            "n_batches" : self.n_batches,
            "seconds" : self.seconds,
        }

    def flat(self, prefix : str = "eval") -> dict:
        out = {f"{prefix}/{k}": v for k, v in self.overall.to_dict().items()}
        for label, m in self.slices.items():
            out[f"{prefix}/slice/{label}/accuracy"] = m.accuracy
            out[f"{prefix}/slice/{label}/accuracy_se"] = m.accuracy_se
            out[f"{prefix}/slice/{label}/n_pairs"] = m.n_pairs
        out[f"{prefix}/seconds"] = self.seconds
        return out


def evaluate(
    model : RewardModel, 
    batches : Iterable[PairBatch], 
    device : str,
    autocast_dtype : Optional[torch.dtype] = None,
    with_slices : bool = True
) -> EvalResult:
    device_type = str(device).split(":")[0]
    amp = torch.autocast(device_type = device_type, dtype = autocast_dtype) if autocast_dtype is not None else nullcontext()
    was_train = model.training
    all_chosen, all_rejected, all_slices = [], [], []
    n_batches = 0
    t0 = time.perf_counter()

    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                b = batch.to(device)
                with amp:
                    rc, rr = model.forward_pair(b)
                all_chosen.append(rc.detach().float().cpu())
                all_rejected.append(rr.detach().float().cpu())
                all_slices.extend(batch.slices)
                n_batches += 1
    finally:
        model.train(was_train)
    
    if n_batches == 0:
        raise ConfigError("evaluate got zero batches")

    all_rc = torch.cat(all_chosen)
    all_rr = torch.cat(all_rejected)
    overall = pair_metrics(all_rc, all_rr)
    slices = slice_metrics(all_rc, all_rr, all_slices) if (with_slices and all_slices) else {}
    return EvalResult(
        overall = overall, 
        slices = slices,
        n_pairs = int(all_rc.shape[0]), 
        n_batches = n_batches,
        seconds = time.perf_counter() - t0
    )
