from __future__ import annotations
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from rlhf.reward_model.core.contracts import ConfigError, DataError
from rlhf.reward_model.dataset.loaders import iter_records, sha256_of

Source = Union[str, Path]
SplitMethod = Literal["hash", "random"]

@dataclass
class SplitReport:
    n_rows_in : int = 0
    n_prompts_in : int = 0
    n_exact_duplicate_rows : int = 0
    n_rows_dropped_as_duplicates : int = 0
    method : str = ""
    val_frac_requested: float = 0.0
    seed : Optional[int] = None
    prompt_field : str = "prompt"
    n_rows_train : int = 0
    n_rows_val : int = 0
    n_prompts_train : int = 0
    n_prompts_val : int = 0
    prompt_frac_val : float = 0.0
    row_frac_val : float = 0.0
    n_overlap_prompts : int = 0
    rows_per_prompt_min : int = 0
    rows_per_prompt_max : int = 0
    
    train_path : str = ""
    val_path : str = ""
    train_sha256 : str = ""
    val_sha256 : str = ""
    inputs : list = field(default_factory = list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

def _hash_unit(prompt : str) -> float:
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[: 8], 16) / (2 ** 32)

def assign_prompts(
    prompts : Sequence[str],
    val_frac : float,
    method : SplitMethod = "hash",
    seed : int = 0,
) -> dict[str, bool]:
    if not 0.0 < val_frac < 1.0:
        raise ConfigError(f"val_frac must be in (0, 1), got {val_frac}")
    
    un_prompts = sorted(set(prompts))
    if len(un_prompts) < 2:
        raise DataError(f"need at least 2 unique prompts to split, got {len(un_prompts)}")
    
    if method == "hash":
        return {p : (_hash_unit(p) < val_frac) for p in un_prompts}

    if method == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(un_prompts))
        n_val = max(1, min(len(un_prompts) - 1, round(val_frac * len(un_prompts))))
        val_idx = set(order[:n_val].tolist())
        return {p : (i in val_idx) for i, p in enumerate(un_prompts)}
    
    raise ConfigError(f"method must be 'hash' or 'random', got {method!r}")

def split_records(
    records : Sequence[dict],
    val_frac : float = 0.1,
    method : SplitMethod = "hash",
    seed : int = 0,
    prompt_field : str = "prompt",
    drop_exact_duplicates : bool = False,
) -> tuple[list[dict], list[dict], SplitReport]:
    rep = SplitReport(method = method, val_frac_requested = val_frac, seed = (seed if method == "random" else None), prompt_field = prompt_field)

    prompts : list[str] = []
    for i, r in enumerate(records):
        if prompt_field not in r:
            raise DataError(f"row {i} has no {prompt_field!r} field. Cannot assign it to a side")
        prompts.append(str(r[prompt_field]))

    rep.n_rows_in = len(records)
    rep.n_prompts_in = len(set(prompts))

    seen : set = set()
    keep = [True] * len(records)
    for i, r in enumerate(records):
        k = json.dumps(r, sort_keys = True, ensure_ascii = False)
        if k in seen:
            rep.n_exact_duplicate_rows += 1
            if drop_exact_duplicates:
                keep[i] = False
                rep.n_rows_dropped_as_duplicates += 1
        seen.add(k)

    is_val = assign_prompts(prompts, val_frac, method = method, seed = seed)

    train, val = [], []
    for i, r in enumerate(records):
        if not keep[i]:
            continue
        if is_val[prompts[i]]:
            val.append(r)
        else:
            train.append(r)

    rep.n_rows_train, rep.n_rows_val = len(train), len(val)
    tr_prompts = {str(r[prompt_field]) for r in train}
    va_prompts = {str(r[prompt_field]) for r in val}
    rep.n_prompts_train, rep.n_prompts_val = len(tr_prompts), len(va_prompts)
    rep.prompt_frac_val = rep.n_prompts_val / max(rep.n_prompts_in, 1)
    rep.row_frac_val = rep.n_rows_val / max(rep.n_rows_train + rep.n_rows_val, 1)

    counts : dict[str, int] = {}
    for p in prompts:
        counts[p] = counts.get(p, 0) + 1
    rep.rows_per_prompt_min = min(counts.values())
    rep.rows_per_prompt_max = max(counts.values())

    rep.n_overlap_prompts = len(tr_prompts & va_prompts)
    if rep.n_overlap_prompts:
        raise DataError(f"prompt-level split produced {rep.n_overlap_prompts} overlapping prompts.")
    return train, val, rep

def _write_records(records : Sequence[dict], path : Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    if path.suffix == ".json":
        path.write_text(json.dumps(list(records), indent = 2, ensure_ascii = False) + "\n")
    elif path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r, ensure_ascii = False) + "\n" for r in records))
    else:
        raise ConfigError(f"can only write .json or .jsonl, got {path.suffix!r} ({path})")

def resplit_files(
    inputs : Sequence[Source],
    train_out : Source,
    val_out : Source,
    val_frac : float = 0.1,
    method : SplitMethod = "hash",
    seed : int = 0,
    prompt_field : str = "prompt",
    drop_exact_duplicates : bool = False,
    json_key : Optional[str] = None,
) -> SplitReport:
    records : list[dict] = []
    rep_inputs : list = []
    for src in inputs:
        src = Path(src)
        records.extend(iter_records(src, json_key = json_key))
        rep_inputs.append((str(src), sha256_of(src)))
    
    if not records:
        raise DataError(f"no records found in inputs: {[str(Path(s)) for s in inputs]}")

    train, val, rep = split_records(
        records, 
        val_frac = val_frac, 
        method = method, 
        seed = seed,
        prompt_field = prompt_field, 
        drop_exact_duplicates = drop_exact_duplicates
    )

    train_out, val_out = Path(train_out), Path(val_out)
    _write_records(train, train_out)
    _write_records(val, val_out)

    rep.inputs = rep_inputs
    rep.train_path, rep.val_path = str(train_out), str(val_out)
    rep.train_sha256 = sha256_of(train_out)
    rep.val_sha256 = sha256_of(val_out)
    return rep

def render(rep : SplitReport, width : int = 76) -> str:
    bar = "=" * width
    L = [bar, "PROMPT LEVEL RE-SPLIT", bar]
    for p, sha in rep.inputs:
        L.append(f"  in   : {p}   sha {sha[:12]}")
    L += [
        f"  rows in         : {rep.n_rows_in:,}   prompts {rep.n_prompts_in:,}"
        f"   rows/prompt {rep.rows_per_prompt_min}..{rep.rows_per_prompt_max}",
        f"  exact dup rows  : {rep.n_exact_duplicate_rows:,}"
        + (f"   (dropped {rep.n_rows_dropped_as_duplicates:,})"
           if rep.n_rows_dropped_as_duplicates else "   (kept)"),
        f"  method          : {rep.method}"
        + (f"   seed {rep.seed}" if rep.seed is not None else "   (seedless, text-stable)"),
        "",
        f"  train           : {rep.n_rows_train:,} rows   {rep.n_prompts_train:,} prompts"
        f"   -> {rep.train_path}   sha {rep.train_sha256[:12]}",
        f"  val             : {rep.n_rows_val:,} rows   {rep.n_prompts_val:,} prompts"
        f"   -> {rep.val_path}   sha {rep.val_sha256[:12]}",
        "",
        f"  val fraction    : prompts {rep.prompt_frac_val:.3f}   rows {rep.row_frac_val:.3f}"
        f"   (requested {rep.val_frac_requested:.3f} of prompts)",
        f"  >> prompt overlap between sides = {rep.n_overlap_prompts}"
        + ("   <-- the number this file exists to make zero" if rep.n_overlap_prompts == 0
           else "   <-- BUG"),
        bar,
    ]
    return "\n".join(L)
