from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union, Literal, TYPE_CHECKING
import torch
from rlhf.core.contracts import ConfigError
from rlhf.core.training.checkpoint import load_checkpoint
from rlhf.reward.model.backbone import Backbone, Pooling
from rlhf.reward.model.model import RewardModel

if TYPE_CHECKING:
    from transformers import AutoTokenizer

PathLike = Union[str, Path]
CheckpointChoices = Literal["latest", "best"]


@dataclass
class ScorerReport:
    run_dir : str = ""
    checkpoint : str = ""
    step : int = 0
    model_name : str = ""
    tokenizer : str = ""
    pooling : str = "mean"
    template : str = "{response}"
    max_length : int = 512
    pad_token_added : bool = False
    device : str = "cpu"
    n_params : int = 0
    n_scored : int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RewardModelScorer:
    def __init__(
        self,
        ckpt_dir : PathLike,
        model_name : str,
        pooling : Pooling,
        template : str,
        tokenizer_name : Optional[str] = None,
        tokenizer : Optional[AutoTokenizer] = None,
        max_length : int = 512,
        head_bias : bool = False,
        device : str = "cpu",
        batch_size : int = 32,
        which : CheckpointChoices = "best",
        name : str = "reward_model",
        run_dir : str = "",
    ):
        from transformers import AutoTokenizer
        if "{response}" not in template:
            raise ConfigError(f"template must contain {{response}} but got {template!r}")

        try:
            template.format(prompt = "", response = "")
        except (KeyError, IndexError, ValueError) as e:
            raise ConfigError(f"template {template!r} does not format with prompt/response: {e}")

        for label, v in (("batch_size", batch_size), ("max_length", max_length)):
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise ConfigError(f"{label} must be an int >= 1, got {v!r}")

        if not isinstance(name, str) or not name:
            raise ConfigError(f"name must be a non-empty string, got {name!r}")

        ckpt = load_checkpoint(ckpt_dir, which = which)
        self.model = RewardModel(Backbone(model_name, pooling = pooling), bias = head_bias)

        try:
            self.model.load_state_dict(ckpt["model"], strict = True)
        except RuntimeError as e:
            raise ConfigError(
                f"checkpoint does not fit the model built from model_name={model_name!r}, "
                f"pooling={pooling!r}, head_bias={head_bias}. read these off the run's "
                f"model.json rather than guessing or use from_run().\n    {e}"
            )

        self.model.eval().to(device)
        self.tokenizer = (
            tokenizer
            if tokenizer
            else (AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else AutoTokenizer.from_pretrained(model_name))
        )
        tok_name = self.tokenizer.name_or_path

        pad_added = False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ConfigError(f"tokenizer {tok_name!r} has neither pad nor eos token, declare a tokenizer that can pad")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_added = True

        self.batch_size = batch_size
        self.device = device
        self.template = template
        self.max_length = max_length
        self.name = name

        self.report = ScorerReport(
            run_dir = str(run_dir),
            checkpoint = str(ckpt_dir),
            step = int(ckpt.get("step", 0)),
            model_name = model_name,
            tokenizer = tok_name,
            pooling = pooling,
            template = template,
            max_length = max_length,
            pad_token_added = pad_added,
            device = str(device),
            n_params = sum(p.numel() for p in self.model.parameters()),
        )

    def __call__(self, prompts : Sequence[str], responses : Sequence[str]):
        if len(prompts) != len(responses):
            raise ConfigError(f"prompts and responses must be the same length, got {len(prompts)} and {len(responses)}")

        texts = [self.template.format(prompt = p, response = r) for p, r in zip(prompts, responses)]
        out : List[float] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tok_out = self.tokenizer(
                batch,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = self.max_length).to(self.device)

            with torch.no_grad():
                model_out = self.model(tok_out["input_ids"], tok_out["attention_mask"])
            out.extend(model_out.flatten().float().cpu().tolist())

        self.report.n_scored += len(out)
        return out


def from_run(
    run_dir : PathLike,
    device : str = "cpu",
    batch_size : int = 32,
    which : str = "best",
    name : Optional[str] = None
) -> RewardModelScorer:

    root = Path(run_dir)
    if not root.is_dir():
        raise ConfigError(f"run directory not found: {root}")

    def read(fname : str) -> dict:
        p = root / fname
        if not p.exists():
            raise ConfigError(
                f"{p} not found. from_run() reads the run's own artifacts; if this "
                f"run predates them, construct RewardModelScorer directly with model_name / pooling / template / max_length"
            )

        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ConfigError(f"{p} is not valid json: {e}")

    def need(cfg, key : str, fname : str):
        if not isinstance(cfg, dict) or key not in cfg:
            raise ConfigError(f"{root / fname} has no {key!r}. from_run() does not guess: a wrong value here changes every score and raises nothing")
        return cfg[key]

    model_cfg = read("model.json")
    collate_cfg = read("collate.json")
    backbone = need(model_cfg, "backbone", "model.json")

    if need(backbone, "custom", "model.json"):
        raise ConfigError(
            f"{root} was trained with a custom backbone, which cannot be rebuilt from json and RewardModelScorer only builds AutoModel backbones. "
            f"build the RewardModel yourself, wrap it in a callable and mark it with @reward_func"
        )

    model_name = need(backbone, "model_name", "model.json")
    if not model_name:
        raise ConfigError(f"{root / 'model.json'} has an empty backbone.model_name")

    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.is_dir():
        raise ConfigError(f"no checkpoints/ directory under {root}")

    return RewardModelScorer(
        ckpt_dir = ckpt_dir,
        model_name = model_name,
        pooling = need(backbone, "pooling", "model.json"),
        template = need(collate_cfg, "template", "collate.json"),
        tokenizer_name = need(collate_cfg, "tokenizer", "collate.json"),
        max_length = int(need(collate_cfg, "max_length", "collate.json")),
        head_bias = bool(need(model_cfg, "head_bias", "model.json")),
        device = device,
        batch_size = batch_size,
        which = which,
        name = name or root.resolve().name,
        run_dir = str(root)
    )


def render(rep : ScorerReport, width : int = 76) -> str:
    bar = "=" * width
    return "\n".join([
        bar, "REWARD MODEL SCORER", bar,
        f"  run           : {rep.run_dir or '<explicit>'}",
        f"  checkpoint    : {rep.checkpoint}   step {rep.step:,}",
        f"  backbone      : {rep.model_name}   pooling {rep.pooling}   {rep.n_params:,} params",
        f"  tokenizer     : {rep.tokenizer}   max_length {rep.max_length}"
        + ("   (pad token added = eos)" if rep.pad_token_added else ""),
        f"  template      : {rep.template!r}",
        f"  device        : {rep.device}   scored {rep.n_scored:,}",
        bar,
    ])
