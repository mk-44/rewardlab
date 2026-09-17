from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Literal
from rlhf.core.config import DataConfig, ExecSection, HubSection, RewardSection, apply_overrides, from_dict, load_yaml
from rlhf.core.contracts import ConfigError
from rlhf.core.device import check_dtype_pair
from rlhf.core.hub import parse_repo_id
from rlhf.core.scoring.rewards import load_reward_functions, resolve_reward_weights
from rlhf.core.training.config import TrainConfig


@dataclass
class PolicyConfig:
    model_name : Optional[str] = "gpt2"
    model_ckpt : Optional[str] = None
    tokenizer : str = ""
    max_length : int = 512
    template : str = "{prompt}\n{response}"
    append_eos : bool = True
    gradient_checkpointing : bool = False

@dataclass
class ReferenceConfig:
    reference_ckpt : Optional[str] = None
    precomputed_logp_path : Optional[str] = None
    precomputed_val_logp_path : Optional[str] = None


@dataclass
class LossConfig:
    beta : float = 0.1
    length_norm : Literal["sum", "mean"] = "sum"
    sft_wt : float = 0.0

@dataclass
class InferenceConfig:
    num_samples_per_prompt : int = 8
    temperature : float = 1.0
    top_p : float = 0.95
    max_new_tokens : int = 32
    eval_prompts : int = 16


@dataclass
class DPOConfig:
    run_name : str = "run"
    out_dir : str = "runs"

    data : DataConfig = field(default_factory = DataConfig)
    policy : PolicyConfig = field(default_factory = PolicyConfig)
    reference : ReferenceConfig = field(default_factory = ReferenceConfig)
    loss : LossConfig = field(default_factory = LossConfig)
    train : TrainConfig = field(default_factory = TrainConfig)
    execution : ExecSection = field(default_factory = ExecSection)
    inference : InferenceConfig = field(default_factory = InferenceConfig)
    reward : RewardSection = field(default_factory = RewardSection)
    hub : HubSection = field(default_factory = HubSection)

    def tokenizer_name(self):
        if self.policy.tokenizer:
            return self.policy.tokenizer    
        return self.policy.model_ckpt or self.policy.model_name or ""

    def ref_fingerprint(self):
        parts = [
            self.tokenizer_name(),
            self.policy.template,
            str(self.policy.max_length),
            str(self.policy.append_eos),
            self.loss.length_norm,
            self.reference.reference_ckpt or "<policy_clone_at_init>",
            self.execution.weights_dtype,
            self.execution.compute_dtype
        ]
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[: 16]


def validate(cfg : DPOConfig) -> None:
    p, l, i = cfg.policy, cfg.loss, cfg.inference
    if bool(p.model_name) == bool(p.model_ckpt):
        raise ConfigError(f"policy : set exactly one of model_name / model_ckpt (got model_name={p.model_name!r}, model_ckpt={p.model_ckpt!r})")

    if p.max_length <= 0:
        raise ConfigError(f"policy.max_length must be positive, got {p.max_length}")

    for ph in ("{prompt}", "{response}"):
        if ph not in p.template:
            raise ConfigError(f"policy.template must contain {ph}, got {p.template!r}")

    if p.template.index("{prompt}") > p.template.index("{response}"):
        raise ConfigError(f"policy.template must put {{prompt}} before {{response}}, got {p.template!r} — the completion mask is defined as everything after the prompt section")

    between = p.template.split("{prompt}")[1].split("{response}")[0]
    if not between:
        raise ConfigError(f"policy.template needs a separator between {{prompt}} and {{response}}, got {p.template!r}")

    if l.beta <= 0:
        raise ConfigError(f"loss.beta must be positive, got {l.beta}")
    if l.sft_wt < 0:
        raise ConfigError(f"loss.sft_wt must be >= 0, got {l.sft_wt}")

    if i.num_samples_per_prompt < 1:
        raise ConfigError(f"inference.num_samples_per_prompt must be >= 1, got {i.num_samples_per_prompt}")
    if i.temperature < 0:
        raise ConfigError(f"inference.temperature must be >= 0 with 0 meaning greedy decoding, got {i.temperature}")
    if i.temperature == 0 and i.num_samples_per_prompt > 1:
        raise ConfigError(f"inference.temperature=0 is greedy decoding so every sample would be the same text, set inference.num_samples_per_prompt to 1 (got {i.num_samples_per_prompt})")
    if not (0.0 < i.top_p <= 1.0):
        raise ConfigError(f"inference.top_p must be in (0, 1], got {i.top_p}")
    if i.max_new_tokens < 1:
        raise ConfigError(f"inference.max_new_tokens must be >= 1, got {i.max_new_tokens}")
    if i.eval_prompts < 0:
        raise ConfigError(f"inference.eval_prompts must be >= 0 (0 disables generation eval), got {i.eval_prompts}")

    if not cfg.train.disable_dropout:
        raise ConfigError(
            "train.disable_dropout must be true for DPO: with dropout on, "
            "log pi(y|x) is stochastic, Delta is noisy, and precomputed reference "
            "log-probs no longer match the policy forward pass"
        )

    check_dtype_pair(cfg.execution.weights_dtype, cfg.execution.compute_dtype)

    ref = cfg.reference
    if ref.precomputed_logp_path and ref.precomputed_val_logp_path and Path(ref.precomputed_logp_path).resolve() == Path(ref.precomputed_val_logp_path).resolve():
        raise ConfigError(
            "reference : precomputed_logp_path and precomputed_val_logp_path are the same file "
            f"{ref.precomputed_logp_path!r}; the val cache would be refused by the data digest "
            "guard only after the train cache had been built"
        )

    r = cfg.reward
    if r.reward_functions:
        resolve_reward_weights(
            load_reward_functions(r.reward_functions),
            r.reward_wts or None,
            r.normalize_reward_func_wts,
        )
    elif r.reward_wts:
        raise ConfigError(
            f"reward.reward_wts has {len(r.reward_wts)} entries but "
            f"reward.reward_functions is empty"
        )

    if cfg.hub.repo_id:
        parse_repo_id(cfg.hub.repo_id)


def load_dpo(path : Optional[str] = None, overrides : Sequence[str] = ()) -> DPOConfig:
    cfg = from_dict(DPOConfig, load_yaml(path) if path else {})
    apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg
