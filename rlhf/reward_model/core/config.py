from __future__ import annotations
import difflib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, get_args, get_origin, get_type_hints
import yaml
from rlhf.reward_model.core.contracts import ConfigError

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}

def coerce(value : Any, target, path : str) -> Any:
    origin = get_origin(target)

    if origin is Union:
        args = [a for a in get_args(target) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return coerce(value, args[0], path)
        raise ConfigError(f"{path}: unsupported Union type {target}")

    if origin is Literal:
        allowed = get_args(target)
        if value in allowed:
            return value
        raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")

    if origin in (list, tuple) or target in (list, tuple):
        kind = origin or target
        if isinstance(value, str):
            value = [v for v in value.split(",") if v]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        inner = get_args(target)[0] if get_args(target) else str
        out = [coerce(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(out) if kind is tuple else out

    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUE:
                return True
            if v in _FALSE:
                return False
        raise ConfigError(f"{path}: cannot read {value!r} as bool, use true/false")

    if target is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: got a bool for an int field")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ConfigError(f"{path}: {value} is not an integer")
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                raise ConfigError(f"{path}: cannot read {value!r} as int")
        raise ConfigError(f"{path}: cannot read {type(value).__name__} as int")

    if target is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: got a bool for a float field")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{path}: cannot read {value!r} as float")
        raise ConfigError(f"{path}: cannot read {type(value).__name__} as float")

    if target is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string, got {type(value).__name__} ({value!r})")

    raise ConfigError(f"{path}: unsupported field type {target}")


def from_dict(cls, d : Optional[dict], path : str = ""):
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping, got {type(d).__name__}")

    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    kwargs = {}

    for key, value in d.items():
        full = f"{path}.{key}" if path else key
        if key not in known:
            hint = difflib.get_close_matches(key, known, n=1)
            suggest = f" — did you mean {hint[0]!r}?" if hint else ""
            raise ConfigError(f"unknown config key {full!r}{suggest} (known keys: {sorted(known)})")
        target = hints[known[key].name]
        
        if is_dataclass(target if isinstance(target, type) else None):
            kwargs[key] = from_dict(target, value, full)
        else:
            kwargs[key] = coerce(value, target, full)

    return cls(**kwargs)

@dataclass
class DataConfig:
    train_path : str = ""
    val_path : str = ""
    format : Literal["pairwise", "kway_ranking", "kway_scores", "arena"] = "pairwise"
    json_key : Optional[str] = None
    slice_keys : tuple = ()
    max_drop_rate : float = 0.05
    prompt_field : str = "prompt"


@dataclass
class SplitConfig:
    val_frac : float = 0.10
    method : Literal["hash", "random"] = "hash"
    seed : int = 0
    drop_exact_duplicates : bool = False


@dataclass
class AuditConfig:
    expected_ceiling : float = 0.75
    seed : int = 0


@dataclass
class EmbedSection:
    backend: Literal["hf", "openai"] = "hf"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    pooling: Literal["mean", "cls"] = "mean"
    batch_size: int = 64
    max_length: int = 512
    cache_dir: str = ".cache/embeddings"


@dataclass
class ProfileConfig:
    embed: EmbedSection = field(default_factory=EmbedSection)
    max_samples: int = 20_000
    seed: int = 0
    eps: float = 0.10
    near_dup_tau: float = 0.95
    min_pairs: int = 100
    min_confident: int = 500
    mmd_n_perm: int = 200
    mmd_max_n: int = 2000
    mmd_min_n: int = 50
    near_dup_max_n: int = 10_000
    offtopic_tau: float = 0.20
    isolated_tau: float = 0.50
    warn_ceiling: float = 0.75
    warn_one_axis_tau: float = 0.35
    warn_controlled_tau: float = 0.05

    def to_embed_config(self):
        from rlhf.reward_model.dataset.distribution import EmbedConfig
        return EmbedConfig(
            backend=self.embed.backend, model=self.embed.model,
            device=self.embed.device, embed_pooling_method=self.embed.pooling,
            batch_size=self.embed.batch_size, max_length=self.embed.max_length,
            cache_dir=self.embed.cache_dir, max_samples=self.max_samples,
            seed=self.seed,
        )

@dataclass
class Config:
    run_name: str = "run"
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)

def load_yaml(path: Union[str, Path]) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(raw).__name__}")
    return raw

def apply_overrides(cfg, overrides: Sequence[str]):
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        obj = cfg
        for part in parts[:-1]:
            if not is_dataclass(obj) or part not in {f.name for f in fields(obj)}:
                raise ConfigError(f"override {dotted!r}: no section {part!r}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        names = {f.name for f in fields(obj)}
        if leaf not in names:
            hint = difflib.get_close_matches(leaf, names, n=1)
            suggest = f" — did you mean {hint[0]!r}?" if hint else ""
            raise ConfigError(f"override {dotted!r}: unknown key{suggest}")
        target = get_type_hints(type(obj))[leaf]
        if is_dataclass(target if isinstance(target, type) else None):
            raise ConfigError(f"override {dotted!r}: {leaf!r} is a section, not a value")
        setattr(obj, leaf, coerce(raw, target, dotted))
    return cfg

def load(path : Optional[Union[str, Path]] = None, overrides : Sequence[str] = ()) -> Config:
    cfg = from_dict(Config, load_yaml(path) if path is not None else {})
    apply_overrides(cfg, overrides)
    return cfg

def to_flat_dict(cfg, prefix : str = "") -> dict:
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(v):
            out.update(to_flat_dict(v, key))
        else:
            out[key] = list(v) if isinstance(v, tuple) else v
    return out

def render(cfg : Config) -> str:
    flat = to_flat_dict(cfg)
    w = max(len(k) for k in flat)
    bar = "=" * 76
    lines = [bar, "RESOLVED CONFIG   (defaults < yaml < cli)", bar]
    lines += [f"  {k:<{w}s} = {json.dumps(v)}" for k, v in flat.items()]
    lines.append(bar)
    return "\n".join(lines)