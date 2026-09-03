from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, get_args
import torch
from torch import nn
from rlhf.reward_model.core.config import ConfigError

Pooling = Literal["last", "mean", "cls"]


def pool_hidden(h : torch.Tensor, mask : torch.Tensor, pooling : Pooling) -> torch.Tensor:
    if h.ndim != 3 or mask.ndim != 2 or (h.shape[: 2] != mask.shape[:]):
        raise ConfigError(f"hidden_dim H shape {h.shape} and mask : {mask.shape} are not as expected.")
    
    if pooling == "mean":
        mask_un = mask.unsqueeze(dim = -1)
        return (h * mask_un.to(h.dtype)).sum(dim = 1) / mask_un.sum(dim = 1).clamp(min = 1e-12)
    elif pooling == "cls":
        return h[:, 0]
    elif pooling == "last":
        last_idxs = (mask.sum(dim = -1) - 1).clamp(min = 0)
        return h[torch.arange(h.shape[0], device = h.device, dtype = torch.int64), last_idxs]
    
    raise ConfigError(f"pooling must be among {', '.join(get_args(Pooling))} only")


@dataclass
class BackboneReport:
    model_name : str = ""
    custom : bool = False
    pooling : str = ""
    hidden_size : int = 0
    n_params : int = 0
    n_trainable : int = 0
    frozen : bool = False
    gradient_checkpointing : bool = False

    def to_dict(self):
        return dict(self.__dict__)


class Backbone(nn.Module):
    def __init__(
        self,
        model_name : str,
        pooling : Pooling,
        custom_model : Optional[nn.Module] = None,
        custom_hidden_size : Optional[int] = None,
        gradient_checkpointing : bool = False,
        freeze : bool = False,
        last_hidden_state_key : Optional[str] = "last_hidden_state"
    ):
        super().__init__()
        if pooling not in get_args(Pooling):
            raise ConfigError(f"pooling must be among the following : {get_args(Pooling)}")
        
        if custom_model is not None:
            if custom_hidden_size is None:
                raise ConfigError("must provide custom_hidden_size if custom_model is not None")
            
            self.model = custom_model
            hidden_size = int(custom_hidden_size)
            custom = True
        else:
            from transformers import AutoModel
            
            self.model = AutoModel.from_pretrained(model_name)
            model_config = self.model.config
            hidden_size = getattr(model_config, "hidden_size", None) or getattr(model_config, "n_embd", None)
            if hidden_size is None:
                raise ConfigError(
                    f"Can not find hidden_size/ n_embd parameter in model config." 
                    f"Pass it as custom_model and custom_hidden_size"
                )
            custom = False
        
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.last_hidden_state_key = last_hidden_state_key

        if gradient_checkpointing:
            grad_check_enable = getattr(self.model, "gradient_checkpointing_enable", None)
            if grad_check_enable is None:
                raise ConfigError(f"{model_name} does not support gradient_checkpointing_enable.")
            grad_check_enable()
        
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        
        self._probe()
        n_params = sum([p.numel() for p in self.model.parameters()])
        n_trainable = sum([p.numel() for p in self.model.parameters() if p.requires_grad])

        self.report = BackboneReport(
            model_name = model_name, 
            custom = custom, 
            pooling = pooling,
            hidden_size = self.hidden_size, 
            n_params = n_params, 
            n_trainable = n_trainable,
            frozen = freeze, 
            gradient_checkpointing = gradient_checkpointing
        )
            


    def _hidden_states(self, input_ids : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids = input_ids, attention_mask = mask)
        if self.last_hidden_state_key is not None:
            h = getattr(out, self.last_hidden_state_key, None)
        else:
            h = out
        if not torch.is_tensor(h):
            raise ConfigError(
                f"Expected {self.last_hidden_state_key} attr in model out as tensor but got {type(h)}" 
                if self.last_hidden_state_key
                else
                f"Expected last_hidden_state as tensor but got {type(h)} "
            )
        return h
    
    def _probe(self):
        batch_size, seq_len = 2, 4
        input_ids = torch.zeros([batch_size, seq_len], dtype = torch.long)
        mask = torch.ones([batch_size, seq_len], dtype = torch.long)
        with torch.no_grad():
            h = self._hidden_states(input_ids = input_ids, mask = mask)
        
        if h.ndim != 3 or h.shape[0] != batch_size or h.shape[1] != seq_len:
            raise ConfigError(f"backbone probe: expected hidden states [2, 4, H], got {tuple(h.shape)}")
        if h.shape[2] != self.hidden_size:
            raise ConfigError(f"backbone probe: hidden size {h.shape[2]} != declared {self.hidden_size}")
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self._hidden_states(input_ids, mask)
        return pool_hidden(h, mask, self.pooling)
