from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

import torch
import yaml

DTYPES = {"float32" : torch.float32, "bfloat16" : torch.bfloat16, "float16" : torch.float16}
RESERVED = ("idx", "response", "n_new_tokens", "finished", "model_name", "repo", "ckpt",
            "max_new_tokens", "batch_size", "template", "step", "best_metric")


def read_jsonl(path : Path) -> List[dict]:
    with open(path, encoding = "utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path : Path, rows : Sequence[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def template_head(template : str) -> str:
    if "{prompt}" not in template or "{response}" not in template:
        raise SystemExit("policy.template must contain {prompt} and {response}")
    return template.split("{response}")[0]


def fetch_checkpoint(repo : str, ckpt : str, download : Optional[Callable] = None) -> Path:
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    return Path(download(repo_id = repo, filename = ckpt))


def load_model(model_name : str, ckpt_path : Optional[Path], device : str, weights_dtype : str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype = DTYPES[weights_dtype])
    meta = {"step" : 0, "best_metric" : None}
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location = "cpu", weights_only = True)
        model.load_state_dict(state["model"], strict = True)
        meta = {"step" : int(state["step"]), "best_metric" : state.get("best_metric")}
        del state
    model.to(device).eval()
    return model, tok, meta


def autocast(device, dtype : Optional[torch.dtype]):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type = torch.device(device).type, dtype = dtype)


def autocast_dtype_for(device : str, weights_dtype : str, compute_dtype : str) -> Optional[torch.dtype]:
    if compute_dtype == weights_dtype or torch.device(device).type == "mps":
        return None
    return DTYPES[compute_dtype]


def stop_ids(model, tok) -> Set[int]:
    ids = model.generation_config.eos_token_id
    ids = [] if ids is None else ([ids] if isinstance(ids, int) else list(ids))
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return {int(i) for i in ids}


def count_new(gen : torch.Tensor, stops : Set[int], pad_id : int) -> Tuple[List[int], List[bool]]:
    halt = stops | {int(pad_id)}
    n_new, finished = [], []
    for row in gen.tolist():
        n, done = len(row), False
        for j, t in enumerate(row):
            if t in halt:
                n, done = j, t in stops
                break
        n_new.append(n)
        finished.append(done)
    return n_new, finished


def generate_batch(model, tok, heads : Sequence[str], max_new_tokens : int, ac_dtype : Optional[torch.dtype]):
    was = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(list(heads), return_tensors = "pt", padding = True, add_special_tokens = False).to(model.device)
        with torch.inference_mode(), autocast(model.device, ac_dtype):
            out = model.generate(**enc, do_sample = False, max_new_tokens = max_new_tokens, pad_token_id = tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1] :]
        text = tok.batch_decode(gen, skip_special_tokens = True)
        n_new, finished = count_new(gen, stop_ids(model, tok), tok.pad_token_id)
        return text, n_new, finished
    finally:
        tok.padding_side = was


@dataclass
class GenReport:
    out : str = ""
    model_name : str = ""
    ckpt : Optional[str] = None
    step : int = 0
    prompts : int = 0
    skipped : int = 0
    generated : int = 0
    finished : int = 0
    mean_new_tokens : float = 0.0
    seconds : float = 0.0


def existing_rows(out : Path, settings : dict) -> Set[int]:
    if not out.exists():
        return set()
    done = set()
    for r in read_jsonl(out):
        for k, v in settings.items():
            if r.get(k) != v:
                raise SystemExit(f"{out} holds rows with {k} = {r.get(k)!r} but this run has {v!r}. use another out path")
        done.add(int(r["idx"]))
    return done


def run(args, download : Optional[Callable] = None) -> GenReport:
    if args.batch_size < 1:
        raise SystemExit("batch size must be >= 1")
    if args.max_new_tokens < 1:
        raise SystemExit("max new tokens must be >= 1")
    with open(args.config, encoding = "utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = args.model_name or cfg["policy"]["model_name"]
    head = template_head(cfg["policy"]["template"])
    weights, compute = cfg["execution"]["weights_dtype"], cfg["execution"]["compute_dtype"]
    ckpt = None if args.ckpt.lower() == "none" else args.ckpt
    out = Path(args.out)
    settings = {
        "model_name" : model_name, "repo" : args.repo, "ckpt" : ckpt,
        "max_new_tokens" : args.max_new_tokens, "batch_size" : args.batch_size,
        "template" : cfg["policy"]["template"],
    }

    rows = read_jsonl(Path(args.prompts))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"{args.prompts} has no rows")
    clash = sorted(set(rows[0]) & set(RESERVED))
    if clash:
        raise SystemExit(f"prompt rows already carry reserved keys {' '.join(clash)}")
    if args.prompt_field not in rows[0]:
        raise SystemExit(f"prompt rows have no field {args.prompt_field!r}")

    done = existing_rows(out, settings)
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    rep = GenReport(out = str(out), model_name = model_name, ckpt = ckpt, prompts = len(rows), skipped = len(done))
    if not todo:
        print(f"{out}: all {len(rows)} rows present. nothing to do", flush = True)
        return rep

    ckpt_path = fetch_checkpoint(args.repo, ckpt, download) if ckpt else None
    model, tok, meta = load_model(model_name, ckpt_path, args.device, weights)
    ac_dtype = autocast_dtype_for(args.device, weights, compute)
    rep.step = meta["step"]
    print(f"{model_name} + {ckpt or 'no checkpoint'}   step {meta['step']}   {len(todo)} of {len(rows)} prompts to go   "
          f"device {args.device}   weights {weights}   autocast {ac_dtype}   cap {args.max_new_tokens}   batch {args.batch_size}", flush = True)

    t0 = time.perf_counter()
    tokens = 0
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk = todo[b * args.batch_size : (b + 1) * args.batch_size]
        heads = [head.format(prompt = r[args.prompt_field]) for _, r in chunk]
        text, n_new, fin = generate_batch(model, tok, heads, args.max_new_tokens, ac_dtype)
        out_rows = [
            {**r, "idx" : i, "response" : t, "n_new_tokens" : n, "finished" : f, **settings,
             "step" : meta["step"], "best_metric" : meta["best_metric"]}
            for (i, r), t, n, f in zip(chunk, text, n_new, fin)
        ]
        append_jsonl(out, out_rows)
        rep.generated += len(chunk)
        rep.finished += sum(fin)
        tokens += sum(n_new)
        dt = time.perf_counter() - t0
        print(f"  batch {b + 1}/{n_batches}   rows {rep.skipped + rep.generated}/{len(rows)}   "
              f"{tokens / dt if dt > 0 else 0.0:.0f} new tokens per second   {dt / 60:.1f} min", flush = True)

    rep.seconds = time.perf_counter() - t0
    rep.mean_new_tokens = tokens / rep.generated
    return rep


def render(rep : GenReport, width : int = 76) -> str:
    bar = "=" * width
    return "\n".join([
        bar, "GENERATIONS", bar,
        f"  out             : {rep.out}",
        f"  model           : {rep.model_name} + {rep.ckpt or 'no checkpoint'}   step {rep.step}",
        f"  rows            : {rep.prompts} prompts   {rep.skipped} already present   {rep.generated} generated now",
        f"  finished        : {rep.finished} of {rep.generated} hit a stop token before the cap",
        f"  new tokens      : mean {rep.mean_new_tokens:.1f}   {rep.seconds:.0f}s",
        bar,
    ])


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = "greedy generations from a Hub checkpoint for every prompt of a jsonl")
    ap.add_argument("--config", default = "experiments/dpo/qwen3_if_tulu/config.yaml", help = "training yaml. template and dtypes come from here")
    ap.add_argument("--model-name", dest = "model_name", default = None, help = "base model on the hub. default policy.model_name of the config")
    ap.add_argument("--repo", default = "mayankkeshari/rewardlab-runs", help = "runs repo holding the checkpoints")
    ap.add_argument("--ckpt", help = "path inside the runs repo such as dpo/qwen3_if_tulu/beta_0.1/checkpoints/best.pt or none for the base model")
    ap.add_argument("--prompts", default = "data/preference_data/tulu3_if_personas/val.jsonl", help = "jsonl with one prompt per row. every field is passed through")
    ap.add_argument("--prompt-field", dest = "prompt_field", default = "prompt")
    ap.add_argument("--out", help = "jsonl that receives one row per prompt")
    ap.add_argument("--max-new-tokens", dest = "max_new_tokens", type = int, default = 1024)
    ap.add_argument("--batch-size", dest = "batch_size", type = int, default = 16)
    ap.add_argument("--limit", type = int, default = 0, help = "first n prompts only. 0 means all")
    ap.add_argument("--device", default = "cuda")
    args = ap.parse_args(argv)
    if not args.ckpt or not args.out:
        raise SystemExit("ckpt and out are required")
    print(render(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
