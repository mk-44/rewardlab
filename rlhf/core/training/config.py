from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union

from rlhf.core.contracts import ConfigError


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
    resume_state : Literal["latest", "best", "all", "none"] = "latest"

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
