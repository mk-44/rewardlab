from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from rlhf.core.config import DataConfig, apply_overrides, from_dict, load_yaml

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
        from rlhf.core.data_analysis.distribution import EmbedConfig
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

def load(path : Optional[Union[str, Path]] = None, overrides : Sequence[str] = ()) -> Config:
    cfg = from_dict(Config, load_yaml(path) if path is not None else {})
    apply_overrides(cfg, overrides)
    return cfg
