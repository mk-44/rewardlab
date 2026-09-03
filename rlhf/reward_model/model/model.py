from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from rlhf.reward_model.core.contracts import ConfigError
from rlhf.reward_model.dataset.collate import PairBatch
from rlhf.reward_model.model.backbone import Backbone
from rlhf.reward_model.model.head import RewardHead


@dataclass
class RewardModelReport:
    backbone: dict
    head_hidden_size: int = 0
    head_bias: bool = False
    n_params: int = 0
    n_trainable: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RewardModel(nn.Module):
    def __init__(self, backbone : Backbone, head : Optional[RewardHead] = None, bias : bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head if head is not None else RewardHead(hidden_size = backbone.hidden_size, bias = bias)

        if self.head.hidden_size != backbone.hidden_size:
            raise ConfigError(
                f"head hidden_size {self.head.hidden_size} != backbone hidden_size {backbone.hidden_size}"
            )
        
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.report = RewardModelReport(
            backbone = backbone.report.to_dict(),
            head_hidden_size = self.head.hidden_size,
            head_bias = self.head.bias,
            n_params = n_params,
            n_trainable = n_trainable
        )
    
    def forward(self, input_ids : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(input_ids, mask))

    def forward_pair(self, batch : PairBatch) -> tuple:
        r_chosen = self.forward(batch.chosen_input_ids, batch.chosen_mask)
        r_rejected = self.forward(batch.rejected_input_ids, batch.rejected_mask)
        return r_chosen, r_rejected
