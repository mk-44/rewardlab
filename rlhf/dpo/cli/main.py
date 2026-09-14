from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING
from rlhf.core.contracts import ConfigError, DataError
from rlhf.core.hub import hub_path_for, preflight, pull_run, push_run, render as render_hub, render_preflight
from rlhf.dpo.config import DPOConfig, load_dpo

if TYPE_CHECKING:
    from rlhf.core.device import ExecutionPlan
    from transformers import PreTrainedModel
    from rlhf.dpo.collate import DPOCollator
    from rlhf.core.preference.schema import PairView


def _run_dir(cfg : DPOConfig) -> Path:
    return Path(cfg.out_dir) / cfg.run_name


def _write(out_dir : Path, name : str, payload : dict) -> None:
    out_dir.mkdir(parents = True, exist_ok = True)
    (out_dir / name).write_text(json.dumps(payload, indent = 2, default = str) + "\n")


def _hub_path(cfg : DPOConfig) -> str:
    return hub_path_for(cfg.out_dir, cfg.run_name)


def _repo_id(cfg : DPOConfig) -> str:
    if not cfg.hub.repo_id:
        raise ConfigError("hub.repo_id is not set. set it in the yaml or pass the hf repo id flag")
    return cfg.hub.repo_id


def _hub_preflight(cfg : DPOConfig) -> None:
    if cfg.hub.repo_id:
        print(render_preflight(preflight(cfg.hub.repo_id, private = cfg.hub.private)))


def _hub_push(cfg : DPOConfig, action : str) -> int:
    if not cfg.hub.repo_id:
        return 0
    try:
        rep = push_run(_run_dir(cfg), cfg.hub.repo_id, _hub_path(cfg), action = action)
    except Exception as e:
        print(f"hub push failed after {action} finished: {type(e).__name__}: {e}. the run is complete on disk. retry with the push verb and the same config", file = sys.stderr)
        return 2
    print(render_hub(rep))
    return 0


def _require_data(cfg : DPOConfig, *splits : str):
    for split in splits:
        path = getattr(cfg.data, f"{split}_path")
        if not path:
            raise ConfigError(f"data.{split}_path is empty. DPO needs both splits: train for the loss, val for selection")
        if not Path(path).exists():
            raise ConfigError(f"data.{split}_path not found : {path}")
        
    if cfg.policy.model_ckpt and not Path(cfg.policy.model_ckpt).exists():
        raise ConfigError(f"policy.model_ckpt not found: {cfg.policy.model_ckpt}")


def _load_pairs(cfg : DPOConfig, path : str, split : str):
    from rlhf.core.preference.loaders import load_groups
    from rlhf.core.preference.schema import groups_to_pairs

    res = load_groups(
        source = path,
        format = cfg.data.format,
        slice_keys = cfg.data.slice_keys,
        json_key = cfg.data.json_key,
        max_drop_rate = cfg.data.max_drop_rate
    )
    pairs = groups_to_pairs(res.groups, seed = cfg.execution.seed)
    if not pairs:
        raise ConfigError(f"{path} produced no pairs")
    return pairs


def _load_policy(cfg : DPOConfig, plan : ExecutionPlan):    
    from rlhf.core.policy.lm import load_policy
    return load_policy(
        model_name = cfg.policy.model_name or None, 
        model_ckpt = cfg.policy.model_ckpt or None,
        device = plan.device,
        dtype = plan.dtype
    )


def _collator(cfg : DPOConfig):
    from rlhf.dpo.collate import DPOCollator
    return DPOCollator(
        tokenizer_name = cfg.tokenizer_name(),
        max_length = cfg.policy.max_length,
        template = cfg.policy.template,
        append_eos = cfg.policy.append_eos
    )


def _references(
    cfg : DPOConfig, 
    policy : PreTrainedModel, 
    collator : DPOCollator, 
    train_pairs : Sequence[PairView], 
    val_pairs : Sequence[PairView],
    device : str
):
    from rlhf.dpo.reference import make_reference, LiveReference
    ref = make_reference(cfg, policy, collator, train_pairs, device, train = True)
    if isinstance(ref, LiveReference) and not cfg.reference.precomputed_val_logp_path:
        return ref, ref
    val_ref = make_reference(
        cfg,
        policy,
        collator,
        pairs = val_pairs,
        device = device,
        train = False
    )
    return ref, val_ref


def _gen_prompts(pairs : Sequence[PairView], n : int) -> list:
    if n == 0:
        return []

    seen, prompts = set(), []
    for p in pairs:
        if p.prompt not in seen:
            seen.add(p.prompt)
            prompts.append(p.prompt)
        if len(prompts) >= n:
            break
    return prompts


def _data_stamp(cfg : DPOConfig) -> dict:
    from rlhf.core.preference.loaders import sha256_of
    return {
        "train_path" : cfg.data.train_path,
        "val_path" : cfg.data.val_path,
        "train_sha256" : sha256_of(cfg.data.train_path),
        "val_sha256" : sha256_of(cfg.data.val_path),
    }


def cmd_build_cache(cfg : DPOConfig, args) -> int:
    from rlhf.core.device import resolve
    from rlhf.dpo.reference import make_reference, render as render_reference

    ref_cfg = cfg.reference
    if not (ref_cfg.precomputed_logp_path or ref_cfg.precomputed_val_logp_path):
        raise ConfigError("reference.precomputed_logp_path and precomputed_val_logp_path are both unset, nothing to build. train would use a live reference")
    
    _require_data(cfg, *[s for s, p in (("train", ref_cfg.precomputed_logp_path), ("val", ref_cfg.precomputed_val_logp_path)) if p])
    plan = resolve(cfg.execution.device, cfg.execution.dtype, cfg.execution.seed, cfg.execution.deterministic)
    policy, _ = _load_policy(cfg, plan)
    collator = _collator(cfg)

    for train, path, data_path in (
        (True, cfg.reference.precomputed_logp_path, cfg.data.train_path),
        (False, cfg.reference.precomputed_val_logp_path, cfg.data.val_path)
    ):
        split = "train" if train else "val"

        if not path:
            print(f"{split}: no cache path configured, this split will use a live reference")
            continue
    
        if args.force and Path(path).exists():
            Path(path).unlink()
        
        existed = Path(path).exists()
        pairs = _load_pairs(cfg, data_path, split)

        ref = make_reference(cfg, policy, collator, pairs, plan.device, train = train)
        print(f"{split}: {'loaded and verified' if existed else 'built'} {path}   ({ref.report.n_cached:,} pairs)")
        print(render_reference(ref.report))
    
    return 0


def cmd_train(cfg : DPOConfig, args) -> int:
    from rlhf.core.config import to_flat_dict
    from rlhf.core.device import resolve
    from rlhf.core.logging import RunLogger
    from rlhf.core.scoring.rewards import load_reward_functions
    from rlhf.dpo.trainer import DPOTrainer, render as render_train

    _require_data(cfg, "train", "val")
    fns = load_reward_functions(cfg.reward.reward_functions)
    plan = resolve(cfg.execution.device, cfg.execution.dtype, seed = cfg.execution.seed, deterministic = cfg.execution.deterministic)
    train_pairs = _load_pairs(cfg, cfg.data.train_path, "train")
    val_pairs = _load_pairs(cfg, cfg.data.val_path, "val")
    _hub_preflight(cfg)

    logger = RunLogger(cfg.out_dir, cfg.run_name, is_main = plan.dist.is_main, mode = "resume" if args.resume else "new")
    logger.stamp("config", to_flat_dict(cfg))
    logger.stamp("data", _data_stamp(cfg))
    
    policy, policy_rep = _load_policy(cfg, plan)
    collator = _collator(cfg)

    ref, val_ref = _references(cfg, policy, collator, train_pairs, val_pairs, plan.device)
    prompts = _gen_prompts(val_pairs, cfg.inference.eval_prompts)

    trainer = DPOTrainer(
        cfg, 
        policy, 
        collator.tok, 
        collator, 
        train_pairs, 
        val_pairs, 
        ref, 
        val_ref, 
        plan, 
        _run_dir(cfg), 
        logger,
        reward_functions = fns or None,
        gen_prompts = prompts or None,
        policy_report = policy_rep,
    )

    rep = trainer.fit(resume = args.resume)
    logger.close()
    print(render_train(rep))
    print(f"-> {_run_dir(cfg)}")
    return _hub_push(cfg, "train")


def cmd_eval(cfg : DPOConfig, args) -> int:
    from rlhf.core.device import resolve
    from rlhf.core.scoring.rewards import load_reward_functions
    from rlhf.core.training.checkpoint import load_checkpoint, restore
    from rlhf.dpo.evaluate import evaluate, render as render_eval
    from rlhf.dpo.reference import make_reference

    _require_data(cfg, "val")
    plan = resolve(cfg.execution.device, cfg.execution.dtype, seed = cfg.execution.seed, deterministic = cfg.execution.deterministic)
    policy, _ = _load_policy(cfg, plan)
    collator = _collator(cfg)
    val_pairs = _load_pairs(cfg, cfg.data.val_path, "val")

    val_ref = make_reference(cfg, policy, collator, val_pairs, plan.device, train = False)
    src = Path(args.ckpt) if args.ckpt else _run_dir(cfg) / "checkpoints"
    state = load_checkpoint(src, which = args.which)

    restore(state, policy, with_rng = False)

    fns = load_reward_functions(cfg.reward.reward_functions)
    prompts = _gen_prompts(val_pairs, cfg.inference.eval_prompts)
    ebs = cfg.train.eval_batch_size
    batches = [collator(val_pairs[i : i + ebs]) for i in range(0, len(val_pairs), ebs)]
    rep = evaluate(
        cfg, 
        policy, 
        val_ref,
        batches,
        tokenizer = collator.tok if prompts else None,
        prompts = prompts or None,
        reward_functions = fns or None,
        device = plan.device,
        with_slices = True,
        seed = plan.seed,
    )
    payload = {"checkpoint" : str(src), "which" : args.which, "checkpoint_step" : state["step"], **rep.to_dict()}
    _write(_run_dir(cfg), "eval_report.json", payload)
    print(render_eval(rep))
    return _hub_push(cfg, "eval")


def cmd_export(cfg : DPOConfig, args) -> int:
    from rlhf.core.device import resolve
    from rlhf.core.training.checkpoint import load_checkpoint, restore

    _require_data(cfg)
    plan = resolve("cpu", "float32", seed = cfg.execution.seed)
    policy, policy_rep = _load_policy(cfg, plan)
    collator = _collator(cfg)
    src = Path(args.ckpt) if args.ckpt else _run_dir(cfg) / "checkpoints"
    state = load_checkpoint(src, which = args.which)
    restore(state, policy, with_rng = False)

    out = Path(args.out) if args.out else _run_dir(cfg) / "export"
    if out.exists() and any(out.iterdir()) and not args.force:
        raise ConfigError(f"{out} exists and is not empty. pass --force to overwrite it")
    
    policy.save_pretrained(out)
    collator.tok.save_pretrained(out)
    _write(out, "export.json", {
        "checkpoint" : str(src),
        "which" : args.which,
        "step" : state["step"],
        "best_metric" : state["best_metric"],
        "policy_source" : policy_rep.source,
        "run_dir" : str(_run_dir(cfg))
    })
    
    print(f"export: step {state['step']} (best_metric {state['best_metric']}) -> {out}")
    print(f"        next leg: policy.model_ckpt: {out}   policy.model_name: null")
    return _hub_push(cfg, "export")


def cmd_push(cfg : DPOConfig, args) -> int:
    repo_id = _repo_id(cfg)
    _hub_preflight(cfg)
    rep = push_run(_run_dir(cfg), repo_id, _hub_path(cfg), action = "push")
    print(render_hub(rep))
    return 0


def cmd_pull(cfg : DPOConfig, args) -> int:
    rep = pull_run(_repo_id(cfg), _hub_path(cfg), _run_dir(cfg))
    print(render_hub(rep))
    return 0


COMMANDS = {
    "build-cache" : cmd_build_cache,
    "train" : cmd_train,
    "eval" : cmd_eval,
    "export" : cmd_export,
    "push" : cmd_push,
    "pull" : cmd_pull,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog = "rlhf dpo", description = "direct preference optimisation: reference caches, training, evaluation, export")
    sub = p.add_subparsers(dest = "command", required = True)
    for name, fn in COMMANDS.items():
        s = sub.add_parser(name)
        s.add_argument("--config", default = None)
        s.add_argument("--set", dest = "overrides", action = "append", default = [], metavar = "KEY=VALUE")
        s.set_defaults(fn = fn)
    sub.choices["build-cache"].add_argument("--force", action = "store_true")
    sub.choices["train"].add_argument("--resume", action = "store_true")
    for verb in ("eval", "export"):
        sub.choices[verb].add_argument("--ckpt", default = None)
        sub.choices[verb].add_argument("--which", default = "best", choices = ("best", "latest"))
    sub.choices["export"].add_argument("--out", default = None)
    sub.choices["export"].add_argument("--force", action = "store_true")
    for verb in ("train", "eval", "export", "push", "pull"):
        sub.choices[verb].add_argument("--hf-repo-id", dest = "hf_repo_id", default = None)
    return p


def main(argv : Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "hf_repo_id", None):
        args.overrides = [*args.overrides, f"hub.repo_id={args.hf_repo_id}"]
    try:
        cfg = load_dpo(args.config, args.overrides)
        return args.fn(cfg, args)
    except (ConfigError, DataError) as e:
        print(f"config error: {e}", file = sys.stderr)
        return 2
