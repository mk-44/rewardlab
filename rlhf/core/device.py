from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch

from rlhf.core.contracts import ConfigError

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_WEIGHTS_DTYPES = ("float32", "bfloat16")


def resolve_device(name: str) -> torch.device:
    base = name.split(":")[0]
    if base not in ("cpu", "cuda", "mps"):
        raise ConfigError(f"device must be cpu, cuda[:N] or mps, got {name!r}")

    if base == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError(
                f"device={name!r} declared but CUDA is not available on this "
                f"machine. Fix the config (device=cpu/mps) or run where CUDA exists. "
                f"No silent fallback."
            )
        if ":" in name:
            idx = int(name.split(":")[1])
            if idx >= torch.cuda.device_count():
                raise ConfigError(
                    f"device={name!r} but only {torch.cuda.device_count()} "
                    f"CUDA device(s) exist (valid: 0..{torch.cuda.device_count() - 1})"
                )
    if base == "mps":
        if not torch.backends.mps.is_available():
            raise ConfigError(
                f"device='mps' declared but MPS is not available "
                f"(built: {torch.backends.mps.is_built()}). No silent fallback."
            )
    return torch.device(name)


def probe_dtype(device: torch.device, dtype: torch.dtype) -> tuple[bool, str]:
    try:
        a = torch.ones((2, 2), dtype=dtype, device=device)
        b = (a @ a).float().sum().item()
        if b != 8.0:
            return False, f"matmul returned {b}, expected 8.0"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def probe_grad_scaler(device_type: str) -> bool:
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=True)
        scaled = scaler.scale(torch.ones(1, device=device_type))
        return bool(torch.isfinite(scaled).all().item())
    except Exception:
        return False


def dtype_from_name(name: str, label: str = "dtype") -> torch.dtype:
    if name not in _DTYPES:
        raise ConfigError(f"{label} must be one of {sorted(_DTYPES)}, got {name!r}")
    return _DTYPES[name]


def resolve_dtype(name: str, device: torch.device, label: str = "dtype") -> torch.dtype:
    dtype = dtype_from_name(name, label)
    ok, why = probe_dtype(device, dtype)
    if not ok:
        raise ConfigError(
            f"{label}={name!r} declared but {device.type} failed the probe: {why}. "
            f"Fix the config or the machine. No silent fallback."
        )
    return dtype


@dataclass
class AmpPolicy:
    autocast: bool
    autocast_dtype: Optional[str]
    grad_scaler: bool


def amp_policy(dtype_name: str) -> AmpPolicy:
    if dtype_name == "float32":
        return AmpPolicy(autocast=False, autocast_dtype=None, grad_scaler=False)
    if dtype_name == "bfloat16":
        return AmpPolicy(autocast=True, autocast_dtype="bfloat16", grad_scaler=False)
    if dtype_name == "float16":
        return AmpPolicy(autocast=True, autocast_dtype="float16", grad_scaler=True)
    raise ConfigError(f"no amp policy for dtype {dtype_name!r}")


def check_dtype_pair(weights_dtype: str, compute_dtype: str) -> None:
    if weights_dtype not in _WEIGHTS_DTYPES:
        raise ConfigError(
            f"execution.weights_dtype must be float32 or bfloat16, got {weights_dtype!r}. "
            f"float16 weights overflow and underflow under AdamW. For mixed precision keep "
            f"weights_dtype float32 and set compute_dtype."
        )
    dtype_from_name(compute_dtype, "execution.compute_dtype")
    if weights_dtype == "bfloat16" and compute_dtype != "bfloat16":
        raise ConfigError(
            f"execution.compute_dtype={compute_dtype!r} cannot differ from execution.weights_dtype='bfloat16'. "
            f"Autocast never widens bfloat16 weights, so bfloat16 weights need compute_dtype bfloat16."
        )


def amp_context(device: Union[str, torch.device], autocast_dtype: Optional[torch.dtype]):
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)


@dataclass
class DistInfo:
    is_dist: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def dist_info(env: Optional[dict] = None) -> DistInfo:
    e = os.environ if env is None else env
    if "RANK" not in e or "WORLD_SIZE" not in e:
        return DistInfo()
    return DistInfo(
        is_dist=int(e["WORLD_SIZE"]) > 1,
        rank=int(e["RANK"]),
        local_rank=int(e.get("LOCAL_RANK", 0)),
        world_size=int(e["WORLD_SIZE"]),
    )


def device_for_rank(base: str, info: DistInfo) -> str:
    if base.split(":")[0] != "cuda" or not info.is_dist:
        return base
    if ":" in base:
        raise ConfigError(
            f"device={base!r} with DDP: do not pin an index in the config, "
            f"the local rank decides it. Declare device=cuda."
        )
    return f"cuda:{info.local_rank}"


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)


@dataclass
class ExecutionPlan:
    requested_device: str = "cpu"
    device: str = "cpu"
    weights_dtype: str = "float32"
    compute_dtype: str = "float32"
    amp: AmpPolicy = field(default_factory=lambda: amp_policy("float32"))
    grad_scaler_available: bool = True
    dist: DistInfo = field(default_factory=DistInfo)
    seed: int = 0
    deterministic: bool = False

    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def torch_weights_dtype(self) -> torch.dtype:
        return _DTYPES[self.weights_dtype]

    def torch_compute_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    def torch_autocast_dtype(self) -> Optional[torch.dtype]:
        return _DTYPES[self.compute_dtype] if self.amp.autocast else None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["amp"] = dict(self.amp.__dict__)
        d["dist"] = dict(self.dist.__dict__)
        d["is_main"] = self.dist.is_main
        return d


def resolve(device: str = "cpu", compute_dtype: str = "float32", seed: int = 0,
            deterministic: bool = False, env: Optional[dict] = None,
            weights_dtype: str = "float32") -> ExecutionPlan:
    info = dist_info(env)
    mapped = device_for_rank(device, info)
    dev = resolve_device(mapped)
    check_dtype_pair(weights_dtype, compute_dtype)
    resolve_dtype(weights_dtype, dev, "weights_dtype")
    resolve_dtype(compute_dtype, dev, "compute_dtype")
    amp = amp_policy(compute_dtype)

    scaler_ok = probe_grad_scaler(dev.type)
    if amp.grad_scaler and not scaler_ok:
        raise ConfigError(
            f"compute_dtype={compute_dtype!r} needs a grad scaler but torch.amp.GradScaler "
            f"does not support device type {dev.type!r}. Use bfloat16 or "
            f"float32 on this device."
        )

    seed_everything(seed, deterministic)
    return ExecutionPlan(
        requested_device=device, device=str(dev),
        weights_dtype=weights_dtype, compute_dtype=compute_dtype, amp=amp,
        grad_scaler_available=scaler_ok, dist=info,
        seed=seed, deterministic=deterministic,
    )


def render(plan: ExecutionPlan, width: int = 76) -> str:
    bar = "=" * width
    a, d = plan.amp, plan.dist
    L = [bar, "EXECUTION PLAN", bar,
         f"  device        : {plan.device}"
         + (f"   (requested {plan.requested_device})" if plan.device != plan.requested_device else ""),
         f"  weights_dtype : {plan.weights_dtype}   probe passed",
         f"  compute_dtype : {plan.compute_dtype}   probe passed",
         f"  amp           : autocast={'on ' + str(a.autocast_dtype) if a.autocast else 'off'}"
         f"   grad_scaler={'yes' if a.grad_scaler else 'no'}",
         f"  dist          : {'rank ' + str(d.rank) + '/' + str(d.world_size) + ' local ' + str(d.local_rank) if d.is_dist else 'single process'}"
         + ("   [main]" if d.is_main else ""),
         f"  seed          : {plan.seed}   deterministic={plan.deterministic}",
         bar]
    return "\n".join(L)
