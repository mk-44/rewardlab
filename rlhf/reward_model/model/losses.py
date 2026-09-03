from __future__ import annotations
from typing import Optional, Union, Literal
import torch
import torch.nn.functional as F
from rlhf.reward_model.core.contracts import ConfigError


def bt_loss(
    r_chosen : torch.Tensor,
    r_rejected : torch.Tensor,
    margin : Optional[Union[float, torch.Tensor]] = None,
    reduction : Literal["mean", "sum", "none"] = "mean"
):
    if reduction not in ("mean", "sum", "none"):
        raise ConfigError(f"reduction has to be among 'mean', 'sum' or 'none' but instead got {reduction}")
    if r_chosen.ndim != 1 or r_rejected.ndim != 1 or r_chosen.shape[0] != r_rejected.shape[0]:
        raise ConfigError(f"Expected r_chosen and r_rejected of shape [B] but instead got r_chosen : {r_chosen.shape}, r_rejected : {r_rejected.shape}")
    
    delta = r_chosen.float() - r_rejected.float()
    if margin is not None:
        m = torch.as_tensor(margin, dtype = torch.float32, device = r_chosen.device)
        if m.dim() not in (0, 1) or (m.dim() == 1 and m.shape != delta.shape):
            raise ConfigError(f"margin must be a scalar or [B]={tuple(delta.shape)}, got {tuple(m.shape)}")
        if bool((m < 0).any()):
            raise ConfigError("margin must be >= 0 (demand a win, not a loss)")
        delta = delta - m
    
    per_pair = F.softplus(-delta)

    if reduction == "none":
        return per_pair
    if reduction == "sum":
        return per_pair.sum()
    return per_pair.mean()
