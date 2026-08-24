from __future__ import annotations
from typing import Optional, Sequence, Union
import torch

class RLHFError(Exception):
    "Root Error Node"

class ShapeError(RLHFError):
    "Shape / dim mismatch error"

class DtypeError(RLHFError):
    "DType mismatch error"

class DeviceError(RLHFError):
    "2 tensors on diff device error"

class ValueRangeError(RLHFError):
    "Value out of allowed range"

class NumericalError(RLHFError):
    "Got Nan / Inf"

class ContractError(RLHFError):
    "Model / protocol broke their promise"

class ConfigError(RLHFError):
    "Wrong / missing value in config"

class DataError(RLHFError):
    "Data struct issues"


# internal_helpers
def _brief(v : object, limit : int = 100) -> str:
    "repr, truncated — but the real length stays visible."
    s = repr(v)
    return s if len(s) <= limit else s[:limit] + f"... ({len(s)} chars)"

def _check_is_tensor(t : object, name : str) -> None:
    "Catch a list/ndarray here, before .dim() throws something confusing."
    if not isinstance(t, torch.Tensor):
        raise ContractError(f"[{name}] Expected a torch.Tensor, got {type(t).__name__}. Value : {_brief(t)}")

def _normalize_device(device : torch.device) -> torch.device:
    "cuda and cuda:0 are one device but compare unequal. Normalise first."
    if device.type == "cuda" and device.index is None:
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", idx)
    return device


# assertions

def assert_rank(t : torch.Tensor, rank : int, name : str) -> None:
    "Number of dimensions."
    _check_is_tensor(t, name)
    if t.dim() != rank:
        raise ShapeError(f"[{name}] Rank mismatch. Expected {rank}D, got {t.dim()}D. Shape: {tuple(t.shape)}.")

def assert_shape(t : torch.Tensor, expected : Sequence[Optional[int]], name : str) -> None:
    "Full shape. None = wildcard dim, e.g. (None, 768) fixes only D."
    _check_is_tensor(t, name)
    actual_shape = tuple(t.shape)
    expected_shape = tuple(expected)

    if len(actual_shape) != len(expected_shape):
        raise ShapeError(f"[{name}] Rank mismatch. Expected {len(expected_shape)}D {expected_shape}, got {len(actual_shape)}D {actual_shape}.")

    for i, (a, e) in enumerate(zip(actual_shape, expected_shape)):
        if e is not None and a != e:
            raise ShapeError(f"[{name}] Shape mismatch at dim {i}. Expected {e}, got {a}. Full shape: {actual_shape} vs expected {expected_shape}.")

def assert_same_shape(a : torch.Tensor, b : torch.Tensor, names : Sequence[str]) -> None:
    "Two tensors must match — (values, returns), (log_probs_new, log_probs_old)."
    _check_is_tensor(a, names[0])
    _check_is_tensor(b, names[1])
    if a.shape != b.shape:
        raise ShapeError(f"Shape mismatch between '{names[0]}' and '{names[1]}'. {names[0]}={tuple(a.shape)}, {names[1]}={tuple(b.shape)}.")

def assert_dtype(t : torch.Tensor, expected : Union[torch.dtype, Sequence[torch.dtype]], name : str) -> None:
    "Exact dtype. `expected` may be one dtype or a tuple of allowed ones."
    _check_is_tensor(t, name)
    allowed = tuple(expected) if isinstance(expected, (tuple, list)) else (expected, )
    if t.dtype not in allowed:
        want = " or ".join(str(d) for d in allowed)
        raise DtypeError(f"[{name}] Dtype mismatch. Expected {want}, got {t.dtype}.")

def assert_float(t : torch.Tensor, name : str) -> None:
    "Any float dtype — fp16 / fp32 / fp64 / bf16. We support 3 of them (plan §5.2)."
    _check_is_tensor(t, name)
    if not t.is_floating_point():
        raise DtypeError(f"[{name}] Expected a floating-point tensor, got {t.dtype}.")

def assert_long(t : torch.Tensor, name : str) -> None:
    "input_ids, attention_mask, indices."
    _check_is_tensor(t, name)
    if t.dtype != torch.long:
        raise DtypeError(f"[{name}] Expected torch.long, got {t.dtype}.")

def assert_bool(t : torch.Tensor, name : str) -> None:
    "Real bool masks — what masking.py will hand out (STEP 5)."
    _check_is_tensor(t, name)
    if t.dtype != torch.bool:
        raise DtypeError(f"[{name}] Expected torch.bool, got {t.dtype}.")

def assert_same_device(*tensors : torch.Tensor, names : Sequence[str]) -> None:
    "All on one device. Prints every name=device, so the odd one out is obvious."
    if len(tensors) != len(names):
        raise ContractError(f"assert_same_device: got {len(tensors)} tensors but {len(names)} names.")

    for t, n in zip(tensors, names):
        _check_is_tensor(t, n)

    devices = [_normalize_device(t.device) for t in tensors]
    if len(set(devices)) > 1:
        detail = ", ".join(f"{n}={d}" for n, d in zip(names, devices))
        raise DeviceError(f"Device mismatch across tensors. Got: {detail}")

def assert_on_device(t : torch.Tensor, device : torch.device, name: str) -> None:
    "On one specific device."
    _check_is_tensor(t, name)
    actual = _normalize_device(t.device)
    want = _normalize_device(torch.device(device))
    if actual != want:
        raise DeviceError(f"[{name}] Expected device {want}, got {actual}.")

def assert_finite(t : torch.Tensor, name : str) -> None:
    "NaN and Inf, counted separately — the causes differ. Expensive: boundary only."
    _check_is_tensor(t, name)
    if not t.is_floating_point():
        return

    if torch.isfinite(t).all():
        return

    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    raise NumericalError(f"[{name}] Non-finite values: {n_nan} NaN, {n_inf} Inf (out of {t.numel()}). Shape: {tuple(t.shape)}, dtype: {t.dtype}.")


def assert_in_range(t : torch.Tensor, lo : Optional[float], hi : Optional[float], name : str) -> None:
    "lo or hi may be None -> one-sided check."
    _check_is_tensor(t, name)
    if t.numel() == 0:
        return

    if lo is not None and (t < lo).any():
        raise ValueRangeError(f"[{name}] Value below lower bound {lo}. Min found: {t.min().item():.6g}.")

    if hi is not None and (t > hi).any():
        raise ValueRangeError(f"[{name}] Value above upper bound {hi}. Max found: {t.max().item():.6g}.")
    
def assert_scalar(t : torch.Tensor, name : str) -> None:
    "0-dim. Catches a (B,) loss before .backward() gives a cryptic error."
    _check_is_tensor(t, name)
    if t.dim() != 0:
        raise ShapeError(f"[{name}] Expected a scalar (0-dim) tensor, got {t.dim()}D with shape {tuple(t.shape)}.")

def assert_loss(t : torch.Tensor, name : str, max_abs : Optional[float] = None) -> None:
    "scalar + float + finite + optional magnitude. max_abs=None disables it — explicitly."
    assert_scalar(t, name)
    assert_float(t, name)
    assert_finite(t, name)
    if max_abs is not None and abs(t.item()) > max_abs:
        raise ValueRangeError(f"[{name}] Loss magnitude {t.item():.4g} exceeds max_abs={max_abs}. Check head init, LR, and gradient clipping.")