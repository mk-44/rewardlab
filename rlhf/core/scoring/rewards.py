from __future__ import annotations
import importlib
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple
from rlhf.core.contracts import ConfigError

CustomRewardFunction = Callable[[Sequence[str], Sequence[str]], Sequence[float]]

@dataclass
class RewardReport:
    n_functions : int = 0
    n_samples : int = 0
    names : tuple = ()
    weights : tuple = ()
    normalized : bool = False

    means : tuple = ()
    mins : tuple = ()
    maxs : tuple = ()

    total_mean : Optional[float] = None
    total_min : Optional[float] = None
    total_max : Optional[float] = None

    reason : Optional[str] = None

    def to_dict(self):
        return dict(self.__dict__)

def reward_function_name(fn : Any) -> str:
    name = getattr(fn, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(fn).__name__


def resolve_reward_weights(
    reward_functions : Optional[Sequence[CustomRewardFunction]],
    reward_wts : Optional[Sequence[float]] = None,
    normalize_reward_func_wts : bool = False,
) -> Tuple[List[CustomRewardFunction], List[float]]:
    """
    Validate the (functions, weights) pair and return them resolved.
    O(n) in the number of functions.
    """

    fns : List[CustomRewardFunction] = list(reward_functions or [])
    for i, fn in enumerate(fns):
        if not callable(fn):
            raise ConfigError(f"reward_functions[{i}] is not callable: {type(fn).__name__}. expected fn(prompts, responses) -> Sequence[float]")

    if reward_wts is None:
        wts = [1.0] * len(fns)
    else:
        wts = [float(w) for w in reward_wts]

    if len(fns) != len(wts):
        raise ConfigError(f"reward_functions and reward_wts must be the same length, got {len(fns)} functions and {len(wts)} weights")

    for i, w in enumerate(wts):
        # Negative weights are legal. Non-finite is not.
        if not math.isfinite(w):
            raise ConfigError(f"reward_wts[{i}] is not finite: {w}")

    if normalize_reward_func_wts and fns:
        total = sum(wts)
        if total == 0.0:
            raise ConfigError("cannot normalize reward_wts: they sum to 0. either drop normalize_reward_func_wts or give the weights a non-zero sum")
        wts = [w / total for w in wts]

    names = [reward_function_name(f) for f in fns]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ConfigError(
            f"reward function names must be unique for the per-function "
            f"breakdown to be readable, got duplicates: {dupes}. set a .name attribute to disambiguate"
        )

    return fns, wts


def combine_rewards(
    prompts : Sequence[str],
    responses : Sequence[str],
    reward_functions : Optional[Sequence[CustomRewardFunction]] = None,
    reward_wts : Optional[Sequence[float]] = None,
    normalize_reward_func_wts : bool = False,
) -> Tuple[Optional[List[float]], RewardReport]:
    """
    Score a batch with every reward function and return the weighted sum.
    O(n_functions * n_samples) plus whatever the functions themselves cost.
    """
    if len(prompts) != len(responses):
        raise ConfigError(f"prompts and responses must be the same length, got {len(prompts)} and {len(responses)}")

    fns, wts = resolve_reward_weights(reward_functions, reward_wts, normalize_reward_func_wts)
    n = len(prompts)

    if not fns:
        return None, RewardReport(
            n_functions = 0, 
            n_samples = n,
            normalized = normalize_reward_func_wts,
            reason = "no reward functions configured",
        )

    names = [reward_function_name(f) for f in fns]
    columns : List[List[float]] = []

    for name, fn in zip(names, fns):
        raw = fn(prompts, responses)
        try:
            col = [float(v) for v in raw]
        except (TypeError, ValueError) as e:
            raise ConfigError(f"reward function {name!r} returned something that is not a sequence of numbers: {type(raw).__name__} ({e})")
        if len(col) != n:
            raise ConfigError(f"reward function {name!r} returned {len(col)} scores for {n} samples. it must return one score per sample, in order")

        for i, v in enumerate(col):
            if not math.isfinite(v):
                raise ConfigError(
                    f"reward function {name!r} returned a non-finite score {v} at index {i}. a NaN here propagates into the advantage and silently kills the update"
                )
        columns.append(col)

    total = [sum(w * col[i] for w, col in zip(wts, columns)) for i in range(n)]

    if n == 0:
        return total, RewardReport(
            n_functions = len(fns),
            n_samples = 0,
            names = tuple(names),
            weights = tuple(wts),
            normalized = normalize_reward_func_wts,
            reason = "empty batch",
        )

    report = RewardReport(
        n_functions = len(fns),
        n_samples = n,
        names = tuple(names),
        weights =tuple(wts),
        normalized = normalize_reward_func_wts,
        means = tuple(sum(c) / n for c in columns),
        mins = tuple(min(c) for c in columns),
        maxs = tuple(max(c) for c in columns),
        total_mean = sum(total) / n,
        total_min = min(total),
        total_max = max(total),
    )
    return total, report

def load_reward_functions(specs : Optional[Sequence[str]]) -> List[CustomRewardFunction]:
    out : List[CustomRewardFunction] = []
    for spec in (specs or []):
        if ":" not in spec:
            raise ConfigError(f"reward function spec {spec!r} must be 'module.path:attribute', e.g. 'custom_functions.scores:my_score'")

        module_path, _, attr = spec.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ConfigError(f"cannot import {module_path!r} for reward function {spec!r}: {e}. is the project root on sys.path?")

        try:
            fn = getattr(module, attr)
        except AttributeError:
            raise ConfigError(f"module {module_path!r} has no attribute {attr!r} (from spec {spec!r})")

        if not callable(fn):
            raise ConfigError(f"{spec!r} resolved to {type(fn).__name__}, which is not callable")
        out.append(fn)
    return out


def reward_func(fn : Optional[Callable] = None, *, name : Optional[str] = None):
    def deco(f):
        if not callable(f):
            raise ConfigError(f"@reward_func expects a callable, got {type(f).__name__}")

        try:
            sig = inspect.signature(f)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            try:
                sig.bind([], [])
            except TypeError as e:
                raise ConfigError(
                    f"reward function {getattr(f, '__name__', f)!r} must be callable as "
                    f"f(prompts, responses), got {sig}: {e}. it is called batched: one "
                    f"list of prompts and one of responses, returning one score per sample"
                )

        if name is not None:
            if not isinstance(name, str) or not name:
                raise ConfigError(f"@reward_func name must be a non-empty string, got {name!r}")
            new_name = name
        elif getattr(f, "name", None):
            return f
        else:
            new_name = getattr(f, "__name__", type(f).__name__)

        try:
            f.name = new_name
        except AttributeError:
            raise ConfigError(
                f"cannot attach .name to {f!r}: bound methods, builtins and __slots__ objects refuse new attributes. wrap it in a def or functools.partial"
            )
        return f

    return deco(fn) if fn is not None else deco


def render(rep : RewardReport, width : int = 76) -> str:
    bar = "=" * width
    L = [bar, "REWARD", bar]
    if rep.n_functions == 0 or rep.n_samples == 0:
        L.append(f"  no reward computed : {rep.reason}")
        L.append(bar)
        return "\n".join(L)

    L.append(f"  samples       : {rep.n_samples:,}")
    L.append(f"  functions     : {rep.n_functions}"
             f"{'   (weights normalized)' if rep.normalized else ''}")
    L.append(f"  {'name':<24}{'weight':>9}{'mean':>11}{'min':>11}{'max':>11}")
    for nm, w, mu, lo, hi in zip(rep.names, rep.weights, rep.means, rep.mins, rep.maxs):
        L.append(f"  {nm[:24]:<24}{w:9.4f}{mu:11.4f}{lo:11.4f}{hi:11.4f}")
    L.append(f"  {'TOTAL':<24}{'':>9}{rep.total_mean:11.4f}{rep.total_min:11.4f}{rep.total_max:11.4f}")
    L.append(bar)
    return "\n".join(L)
