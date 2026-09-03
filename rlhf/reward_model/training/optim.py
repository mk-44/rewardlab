from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import torch
from torch import nn
from rlhf.reward_model.core.contracts import ConfigError


@dataclass
class OptimReport:
    lr : float = 0.0
    weight_decay : float = 0.0
    betas : tuple[float, float] = (0.9, 0.999)
    eps : float = 1e-8
    n_decay_tensors : int = 0
    n_decay_params : int = 0
    n_no_decay_tensors : int = 0
    n_no_decay_params : int = 0
    n_frozen_excluded: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["betas"] = list(d["betas"])
        return d


def build_optimizer(
    model : nn.Module, 
    lr : float, 
    weight_decay : float = 0.01,
    betas : Tuple[float, float] = (0.9, 0.999),
    eps : float = 1e-8
) -> Tuple[torch.optim.AdamW, OptimReport]:
    
    if not (isinstance(lr, float) and math.isfinite(lr) and lr > 0):
        raise ConfigError(f"lr must be a positive finite float, got {lr!r}")
    if not (isinstance(weight_decay, (int, float)) and weight_decay >= 0):
        raise ConfigError(f"weight_decay must be >= 0, got {weight_decay!r}")
    if len(betas) != 2 or not all(0.0 <= float(b) < 1.0 for b in betas):
        raise ConfigError(f"betas must be two floats in [0, 1), got {betas!r}")
    if not (isinstance(eps, float) and eps > 0):
        raise ConfigError(f"eps must be a positive float, got {eps!r}")
    
    decay, no_decay, frozen = [], [], 0
    for p in model.parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
            
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    
    if not decay and not no_decay:
        raise ConfigError("no trainable parameters: everything is frozen. nothing to optimize")
    
    grps = [
        {"params" : decay, "weight_decay" : float(weight_decay)},
        {"params" : no_decay, "weight_decay" : 0.0}
    ]
    opt = torch.optim.AdamW(grps, lr = lr, betas=(float(betas[0]), float(betas[1])), eps = eps)
    report = OptimReport(
        lr = lr, 
        weight_decay = float(weight_decay),
        betas = (float(betas[0]), float(betas[1])), 
        eps = eps,
        n_decay_tensors = len(decay),
        n_decay_params = sum(p.numel() for p in decay),
        n_no_decay_tensors = len(no_decay),
        n_no_decay_params = sum(p.numel() for p in no_decay),
        n_frozen_excluded = frozen
    )
    return opt, report


def schedule_factor(
    step : int,
    kind : Literal["constant", "linear", "cosine"],
    warmup_steps : int,
    total_steps : Optional[int] = None
) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    
    if kind == "constant":
        return 1.0
    
    span = total_steps - warmup_steps
    progress = min(float(max(step - warmup_steps, 0) / span), 1.0)

    if kind == "linear":
        return 1 - progress

    return 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class SchedReport:
    kind : Literal["constant", "linear", "cosine"] = "constant"
    warmup_steps : int = 0
    total_steps : Optional[int] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_scheduler(
    optimizer : torch.optim.Optimizer, 
    kind : Literal["constant", "linear", "cosine"] = "constant",
    warmup_steps : int = 0, 
    total_steps : Optional[int] = None
) -> Tuple[torch.optim.lr_scheduler.LambdaLR, SchedReport]:
    
    if kind not in ("constant", "linear", "cosine"):
        raise ConfigError(f"kind must be constant/linear/cosine, got {kind!r}")

    if not (isinstance(warmup_steps, int) and not isinstance(warmup_steps, bool) and warmup_steps >= 0):
        raise ConfigError(f"warmup_steps must be an int >= 0, got {warmup_steps!r}")
    
    if kind in ("linear", "cosine"):
        if not isinstance(total_steps, int) or isinstance(total_steps, bool):
            raise ConfigError(f"{kind} schedule requires integer total_steps, got {total_steps!r}")
        if total_steps <= warmup_steps:
            raise ConfigError(f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps}), a zero-length decay phase is a config error")
    
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer = optimizer,
        lr_lambda = lambda step : schedule_factor(step, kind, warmup_steps, total_steps)
    )
    return sched, SchedReport(kind=kind, warmup_steps=warmup_steps, total_steps=total_steps)
