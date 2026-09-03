from __future__ import annotations
import torch
from torch import nn
from rlhf.reward_model.core.contracts import ConfigError


class RewardHead(nn.Module):
    def __init__(self, hidden_size : int, bias : bool = False):
        super().__init__()
        if hidden_size <= 0:
            raise ConfigError(f"hidden_size must be > 0, but provided {hidden_size}")
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)
        self.linear = nn.Linear(hidden_size, 1, bias = self.bias)
    
    def forward(self, h : torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or h.shape[1] != self.hidden_size:
            raise ConfigError(f"Expected h shape [B, {self.hidden_size}] but got {h.shape}")
        with torch.autocast(device_type = h.device.type, enabled = False):
            rew = self.linear(h.float())
        return rew.squeeze(-1)
