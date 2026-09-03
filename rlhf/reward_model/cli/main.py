from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from rlhf.reward_model.core.config import (
    AuditConfig, 
    DataConfig, 
    ProfileConfig,
    SplitConfig, 
    apply_overrides, 
    from_dict,
    load_yaml
)

from rlhf.reward_model.core.contracts import ConfigError
from rlhf.reward_model.training.trainer import TrainConfig

Pooling_choices = ("last", "mean", "cls")


@dataclass
class ModelSection:
    backbone : str = "gpt2"
    tokenizer : str = ""
    pooling : str = "last"
    max_length : int = 512
    template : str = "{prompt}\n{response}"
    freeze : bool = False
    bias : bool = False
    pair_policy : str = "all_pairs"
    max_pairs_per_group : Optional[int] = None


@dataclass
class ExecSection:
    device : str = "cpu"
    dtype : str = "float32"
    seed : int = 0
    deterministic : bool = False


@dataclass
class ResplitOut:
    train_out : str = "preference_data/train.jsonl"
    val_out : str = "preference_data/val.jsonl"


@dataclass
class AppConfig:
    run_name : str = "run"
    out_dir : str = "runs"
    reports_dir : str = "preference_data/reports"
    data : DataConfig = field(default_factory = DataConfig)
    split : SplitConfig = field(default_factory = SplitConfig)
    audit : AuditConfig = field(default_factory = AuditConfig)
    profile : ProfileConfig = field(default_factory = ProfileConfig)
    model : ModelSection = field(default_factory = ModelSection)
    execution : ExecSection = field(default_factory = ExecSection)
    train : TrainConfig = field(default_factory = TrainConfig)
    resplit : ResplitOut = field(default_factory = ResplitOut)


def load_app(path : Optional[str] = None, overrides : Sequence[str] = ()) -> AppConfig:
    cfg = from_dict(AppConfig, load_yaml(path) if path else {})
    apply_overrides(cfg, overrides)
    if cfg.model.pooling not in Pooling_choices:
        raise ConfigError(f"model.pooling must be one of {Pooling_choices}, got {cfg.model.pooling!r}")
    return cfg


def _load_pair_views(cfg : AppConfig, path : str):
    from rlhf.reward_model.dataset.loaders import load_groups
    from rlhf.reward_model.dataset.schema import groups_to_pairs
    if not path:
        raise ConfigError("data path is empty. set data.train_path / data.val_path")
    res = load_groups(
        path, 
        cfg.data.format,
        slice_keys = cfg.data.slice_keys, 
        json_key = cfg.data.json_key,
        max_drop_rate = cfg.data.max_drop_rate
    )
    
    pairs = groups_to_pairs(res.groups, cfg.model.pair_policy, cfg.model.max_pairs_per_group, cfg.execution.seed)
    return res.groups, pairs


def _build_collator(cfg: AppConfig):
    from rlhf.reward_model.dataset.collate import PairCollator
    tok = cfg.model.tokenizer or cfg.model.backbone
    return PairCollator(tok, max_length = cfg.model.max_length, template = cfg.model.template)


def _build_store(cfg: AppConfig, path: str, collator):
    from rlhf.reward_model.dataset.collate import PretokenizedPairs
    _, pairs = _load_pair_views(cfg, path)
    return PretokenizedPairs(pairs, collator)


def _build_model(cfg : AppConfig):
    from rlhf.reward_model.model.backbone import Backbone
    from rlhf.reward_model.model.model import RewardModel
    bb = Backbone(cfg.model.backbone, pooling = cfg.model.pooling, freeze = cfg.model.freeze)
    return RewardModel(bb, bias = cfg.model.bias)

def _write(out_dir : Path, name : str, payload : dict) -> None:
    out_dir.mkdir(parents = True, exist_ok = True)
    (out_dir / name).write_text(json.dumps(payload, indent = 2, default = str) + "\n")


def cmd_resplit(cfg : AppConfig, args) -> int:
    from rlhf.reward_model.dataset.splits import resplit_files
    inputs = [p for p in (cfg.data.train_path, cfg.data.val_path) if p]
    
    if not inputs:
        raise ConfigError("resplit needs data.train_path (and optionally data.val_path)")
    
    rep = resplit_files(
        inputs, 
        cfg.resplit.train_out, 
        cfg.resplit.val_out,
        val_frac = cfg.split.val_frac, 
        method = cfg.split.method,
        seed = cfg.split.seed, 
        prompt_field = cfg.data.prompt_field,
        drop_exact_duplicates = cfg.split.drop_exact_duplicates,
        json_key = cfg.data.json_key
    )
    
    _write(Path(cfg.out_dir), "split_report.json", rep.to_dict())
    print(f"resplit: {rep.n_rows_train} train / {rep.n_rows_val} val rows, {rep.n_prompts_train}/{rep.n_prompts_val} prompts -> {cfg.resplit.train_out}, {cfg.resplit.val_out}")
    return 0


def cmd_audit(cfg: AppConfig, args) -> int:
    from rlhf.reward_model.dataset.audit import audit, render
    train_groups, _ = _load_pair_views(cfg, cfg.data.train_path)
    val_groups = None
    
    if cfg.data.val_path:
        val_groups, _ = _load_pair_views(cfg, cfg.data.val_path)
    
    rep = audit(
        train_groups, 
        val_groups, 
        source = cfg.data.train_path,
        seed = cfg.audit.seed, 
        expected_ceiling = cfg.audit.expected_ceiling
    )

    text = render(rep)
    print(text)
    out = Path(cfg.reports_dir)
    _write(out, "audit_report.json", rep.to_dict())
    (out / "audit_report.txt").write_text(text + "\n")
    return 0


def cmd_profile(cfg : AppConfig, args) -> int:
    from rlhf.reward_model.dataset.distribution import profile, render
    train_groups, _ = _load_pair_views(cfg, cfg.data.train_path)
    val_groups = None

    if cfg.data.val_path:
        val_groups, _ = _load_pair_views(cfg, cfg.data.val_path)
    
    rep = profile(
        train_groups, 
        val_groups, 
        cfg = cfg.profile.to_embed_config(),
        source = cfg.data.train_path, 
        eps = cfg.profile.eps,
        tau = cfg.profile.near_dup_tau, 
        seed = cfg.profile.seed
    )
    
    text = render(rep)
    print(text)
    out = Path(cfg.reports_dir)
    _write(out, "profile_report.json", rep.to_dict())

    (out / "profile_report.txt").write_text(text + "\n")
    return 0


def cmd_train(cfg: AppConfig, args) -> int:
    from rlhf.reward_model.core.device import resolve
    from rlhf.reward_model.core.logging import RunLogger
    from rlhf.reward_model.training.trainer import Trainer

    if "{prompt}" not in cfg.model.template:
        print(f"warning: template {cfg.model.template!r} contains no {{prompt}}. The model will score responses without seeing "
              f"their prompts. Deliberate for probes; a mistake for production reward models.", file = sys.stderr)

    audit_p = Path(cfg.reports_dir) / "audit_report.json"
    if not audit_p.exists():
        print(f"warning: no audit report at {audit_p} — training on undiagnosed data", file = sys.stderr)
    else:
        audit_src = json.loads(audit_p.read_text()).get("source")
        if audit_src != cfg.data.train_path:
            print(f"warning: audit report describes {audit_src!r}, not {cfg.data.train_path!r}", file = sys.stderr)

    collator = _build_collator(cfg)
    train_store = _build_store(cfg, cfg.data.train_path, collator)
    val_store = _build_store(cfg, cfg.data.val_path, collator)
    model = _build_model(cfg)
    plan = resolve(cfg.execution.device, cfg.execution.dtype, seed = cfg.execution.seed, deterministic = cfg.execution.deterministic)
    logger = RunLogger(cfg.out_dir, cfg.run_name, is_main = plan.dist.is_main, mode = "resume" if args.resume else "new")
    logger.stamp("collate", collator.report.to_dict())
    from rlhf.reward_model.dataset.loaders import sha256_of
    logger.stamp("data", {
      "train_path" : cfg.data.train_path,
      "val_path" : cfg.data.val_path,
      "train_sha256" : sha256_of(cfg.data.train_path),
      "val_sha256" : sha256_of(cfg.data.val_path)
      })
      
    trainer = Trainer(cfg.train, model, train_store, val_store, plan, logger, Path(cfg.out_dir) / cfg.run_name)
    rep = trainer.fit(resume = args.resume)
    print(f"train: {rep.steps} steps, best={rep.best_metric} @ {rep.best_step}, "
          f"final_acc={rep.final_accuracy:.4f}, "
          f"{'stopped early, ' if rep.early_stopped else ''}"
          f"{rep.wall_seconds:.1f}s -> {Path(cfg.out_dir) / cfg.run_name}")
    return 0


def cmd_eval(cfg : AppConfig, args) -> int:
    from rlhf.reward_model.core.device import resolve
    from rlhf.reward_model.training.checkpoint import load_checkpoint, restore
    from rlhf.reward_model.training.evaluate import evaluate
    val_store = _build_store(cfg, cfg.data.val_path, _build_collator(cfg))
    model = _build_model(cfg)
    src = args.ckpt if args.ckpt else Path(cfg.out_dir) / cfg.run_name / "checkpoints"
    state = load_checkpoint(src, which = args.which)
    restore(state, model)
    plan = resolve(cfg.execution.device, cfg.execution.dtype, seed = cfg.execution.seed)
    model = model.to(plan.device)

    def batches():
        bs = cfg.train.eval_batch_size
        for i in range(0, len(val_store), bs):
            yield val_store.batch(list(range(i, min(i + bs, len(val_store)))))

    result = evaluate(model, batches(), device = plan.device, autocast_dtype = plan.torch_dtype() if plan.amp.autocast else None)
    payload = {"checkpoint_step": state["step"], **result.to_dict()}
    _write(Path(cfg.out_dir) / cfg.run_name, "eval_report.json", payload)
    o = result.overall
    
    print(f"eval @ step {state['step']}: acc={o.accuracy:.4f} (se {o.accuracy_se:.4f}) "
          f"mean_margin={o.mean_margin:+.4f} n={result.n_pairs}")
    return 0


def cmd_ledger(cfg: AppConfig, args) -> int:
    from rlhf.reward_model.analysis.ledger import build_ledger, check_shared_space, render_ledger
    run_dir = Path(cfg.out_dir) / cfg.run_name

    def rd(p):
        p = Path(p)
        if not p.exists():
            raise ConfigError(
                f"missing receipt: {p} — the ledger only reads saved "
                f"reports (tier2_report.json comes from the tier-2 "
                f"analysis notebook)"
            )
        return json.loads(p.read_text())

    audit = rd(args.audit_json or Path(cfg.reports_dir) / "audit_report.json")
    profile = rd(args.profile_json or Path(cfg.reports_dir) / "profile_report.json")
    audit_src = audit.get("source")
    profile_src = profile.get("source")
    for label, src in (("audit", audit_src), ("profile", profile_src)):
        if src != cfg.data.train_path:
            print(f"warning: {label} report describes {src!r}, not "
                  f"{cfg.data.train_path!r} — predicted column may be about "
                  f"different data", file = sys.stderr)
    train_rep = rd(run_dir / "train_report.json")
    eval_rep = rd(run_dir / "eval_report.json")
    tier2 = rd(run_dir / "tier2_report.json")

    shared = check_shared_space(
        rd(run_dir / "model.json"), 
        rd(run_dir / "collate.json"),
        cfg.profile.embed.model, 
        cfg.profile.embed.pooling,
        cfg.profile.embed.max_length
    )
    
    rows = build_ledger(
        audit, 
        profile, 
        train_rep, 
        eval_rep, 
        tier2,
        trainer_stamp = rd(run_dir / "trainer.json"),
        shared_space = shared
    )
    
    text = render_ledger(rows, title=f"{Path(cfg.out_dir).name} / {cfg.run_name}")
    print(text)
    _write(Path(cfg.out_dir), "ledger.json", {"shared_space": shared, "audit_source": audit_src, "profile_source": profile_src, "rows": [r.to_dict() for r in rows]})
    (Path(cfg.out_dir) / "ledger.md").write_text(f"# Ledger — {Path(cfg.out_dir).name} / {cfg.run_name}\n\n```\n{text}\n```\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog = "rewardlab", description = "reward-model lab: data diagnostics and training")
    sub = p.add_subparsers(dest = "command", required = True)
    
    for name, fn in COMMANDS.items():
        s = sub.add_parser(name)
        s.add_argument("--config", default=None)
        s.add_argument("--set", dest="overrides", action="append", default=[],
                       metavar="KEY=VALUE")
        s.set_defaults(fn=fn)
    
    sub.choices["train"].add_argument("--resume", action = "store_true")
    sub.choices["eval"].add_argument("--ckpt", default = None)
    sub.choices["eval"].add_argument("--which", default = "best", choices = ("best", "latest"))
    sub.choices["ledger"].add_argument("--audit-json", default = None)
    sub.choices["ledger"].add_argument("--profile-json", default = None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_app(args.config, args.overrides)
        return args.fn(cfg, args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


COMMANDS = {
    "resplit": cmd_resplit,
    "audit": cmd_audit,
    "profile": cmd_profile,
    "train": cmd_train,
    "eval": cmd_eval,
    "ledger": cmd_ledger,
}
