from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from rlhf.reward_model.core.contracts import ConfigError


@dataclass
class PairMetrics:
    n_pairs: int = 0
    accuracy: float = 0.0
    accuracy_se: float = 0.0
    frac_ties: float = 0.0
    mean_margin: float = 0.0
    p10_margin: float = 0.0
    median_margin: float = 0.0
    p90_margin: float = 0.0
    mean_chosen: float = 0.0
    mean_rejected: float = 0.0
    abs_reward_max: float = 0.0
    mean_loss: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def pair_metrics(r_chosen : torch.Tensor, r_rejected : torch.Tensor) -> PairMetrics:
    if r_chosen.ndim != 1 or r_rejected.ndim != 1 or r_chosen.shape[0] != r_rejected.shape[0]:
        raise ConfigError(f"Expected r_chosen and r_rejected of shape [B] but instead got r_chosen : {r_chosen.shape}, r_rejected : {r_rejected.shape}")
    
    if r_chosen.numel() == 0:
        raise ConfigError("pair_metrics on zero pairs")
    
    rc = r_chosen.detach().float().cpu()
    rr = r_rejected.detach().float().cpu()
    delta = rc - rr
    n = delta.numel()
    
    acc = float(((delta > 0).float() + 0.5 * (delta == 0).float()).mean())
    q = torch.quantile(delta, torch.tensor([0.10, 0.50, 0.90], device = rc.device))
    
    return PairMetrics(
        n_pairs = n,
        accuracy = acc,
        accuracy_se = float((acc * (1.0 - acc) / n) ** 0.5),
        frac_ties = float((delta == 0).float().mean()),
        mean_margin = float(delta.mean()),
        p10_margin = float(q[0]),
        median_margin = float(q[1]),
        p90_margin = float(q[2]),
        mean_chosen = float(rc.mean()),
        mean_rejected = float(rr.mean()),
        abs_reward_max = float(torch.maximum(rc.abs().max(), rr.abs().max())),
        mean_loss = float(F.softplus(-delta).mean())
    )
