from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import re
import string
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pronouncing
import torch

from rlhf.core.policy.lm import load_policy, sequence_logprobs
from rlhf.core.preference.loaders import load_groups
from rlhf.core.preference.schema import groups_to_pairs


_PUNCT = str.maketrans("", "", string.punctuation)


def last_word(line : str) -> str:
    words = line.translate(_PUNCT).lower().split()
    return words[-1] if words else ""


def rhyme_key(word : str) -> str:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return ""
    return re.sub(r"\d", "", pronouncing.rhyming_part(phones[0]))


def couplet_lines(text : str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def couplet_rhymes(text : str) -> bool:
    lines = couplet_lines(text)
    if len(lines) < 2:
        return False
    a, b = last_word(lines[0]), last_word(lines[1])
    if not a or not b or a == b:
        return False
    ka, kb = rhyme_key(a), rhyme_key(b)
    return bool(ka) and ka == kb


def common_suffix(a : str, b : str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def near_miss_bucket(n : int) -> str:
    return "3+" if n >= 3 else str(n)


def split_last(line : str) -> Tuple[str, str]:
    toks = line.split()
    m = re.match(r"^(.*?)([^A-Za-z0-9]*)$", toks[-1])
    return " ".join(toks[: -1]), (m.group(2) if m else "")


def replace_end(lines : Sequence[str], word : str) -> str:
    prefix, punct = split_last(lines[1])
    line2 = (prefix + " " + word + punct) if prefix else (word + punct)
    return lines[0] + "\n" + line2


@dataclass
class Pool:
    words : List[str]
    log2 : List[float]
    count : Dict[str, int]
    key : Dict[str, str]


def build_pool(rows : Sequence[dict]) -> Pool:
    count : collections.Counter = collections.Counter()
    for r in rows:
        lines = couplet_lines(r["chosen"])
        if len(lines) >= 2:
            w = last_word(lines[1])
            if w:
                count[w] += 1
    key = {w : rhyme_key(w) for w in count}
    words = sorted(count, key = lambda w : (math.log2(count[w]), w))
    return Pool(words = words, log2 = [math.log2(count[w]) for w in words], count = count, key = key)


def candidates(pool : Pool, w1 : str, k1 : str, w2 : str, band : float, near_miss : str) -> List[str]:
    c = math.log2(pool.count[w2])
    lo = bisect.bisect_left(pool.log2, c - band)
    hi = bisect.bisect_right(pool.log2, c + band)
    out = []
    for w in pool.words[lo : hi]:
        if w == w1 or w == w2:
            continue
        kw = pool.key[w]
        if not kw or kw == k1:
            continue
        if near_miss == "exclude" and common_suffix(w1, w) >= 2:
            continue
        out.append(w)
    return out


class BaseScorer:
    def __init__(self, model_name : str, template : str, append_eos : bool, device : str, batch_size : int):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model, _ = load_policy(model_name = model_name, device = device, dtype = "float32")
        self.model.eval()
        self.head, self.tail = template.split("{response}")
        self.append_eos = append_eos
        self.device = device
        self.batch_size = batch_size
        self.eos = self.tok.eos_token_id
        self._first : Dict[Tuple[str, bool], int] = {}

    def first_id(self, word : str, after_space : bool) -> int:
        k = (word, after_space)
        if k not in self._first:
            text = (" " + word) if after_space else word
            self._first[k] = self.tok(text, add_special_tokens = False)["input_ids"][0]
        return self._first[k]

    def encode(self, prompt : str, response : str) -> Tuple[List[int], List[int]]:
        p = self.tok(self.head.format(prompt = prompt), add_special_tokens = False)["input_ids"]
        r = self.tok(response + self.tail, add_special_tokens = False)["input_ids"]
        if self.append_eos:
            r = r + [self.eos]
        return p, r

    def _pad(self, seqs : Sequence[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.eos, dtype = torch.long)
        att = torch.zeros((len(seqs), L), dtype = torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s, dtype = torch.long)
            att[j, : len(s)] = 1
        return ids, att, [len(s) - 1 for s in seqs]

    @torch.no_grad()
    def slot_logprobs(self, prompts : Sequence[str], prefixes : Sequence[str], cand_ids : Sequence[List[int]]) -> List[List[float]]:
        out : List[List[float]] = []
        bs = self.batch_size
        for i in range(0, len(prompts), bs):
            seqs = []
            for p, pre in zip(prompts[i : i + bs], prefixes[i : i + bs]):
                h = self.tok(self.head.format(prompt = p), add_special_tokens = False)["input_ids"]
                seqs.append(h + self.tok(pre, add_special_tokens = False)["input_ids"])
            ids, att, last = self._pad(seqs)
            logits = self.model(input_ids = ids.to(self.device), attention_mask = att.to(self.device)).logits
            rows = logits[torch.arange(len(seqs)), torch.tensor(last)].float()
            logp = torch.log_softmax(rows, dim = -1).cpu()
            for j, cids in enumerate(cand_ids[i : i + bs]):
                out.append(logp[j, torch.tensor(cids, dtype = torch.long)].tolist() if cids else [])
        return out

    @torch.no_grad()
    def logps(self, prompts : Sequence[str], responses : Sequence[str]) -> Tuple[List[float], List[float]]:
        total : List[float] = []
        last : List[float] = []
        bs = self.batch_size
        for i in range(0, len(prompts), bs):
            enc = [self.encode(p, r) for p, r in zip(prompts[i : i + bs], responses[i : i + bs])]
            ids, att, ends = self._pad([p + r for p, r in enc])
            comp = torch.zeros_like(ids)
            for j, (p, r) in enumerate(enc):
                comp[j, len(p) : len(p) + len(r)] = 1
            ids, att, comp = ids.to(self.device), att.to(self.device), comp.to(self.device)
            logits = self.model(input_ids = ids, attention_mask = att).logits
            lp = sequence_logprobs(self.model, ids, att, comp, "sum", logits)
            rows = torch.arange(len(enc), device = ids.device)
            pos = torch.tensor(ends, dtype = torch.long, device = ids.device)
            end_logp = torch.log_softmax(logits[rows, pos - 1].float(), dim = -1)[rows, ids[rows, pos]]
            total.extend(lp.float().cpu().tolist())
            last.extend(end_logp.cpu().tolist())
        return total, last


@dataclass
class BuildReport:
    split : str
    rows_in : int = 0
    pool_words : int = 0
    processed : int = 0
    skipped_not_two_lines : int = 0
    skipped_chosen_not_rhyme : int = 0
    skipped_no_candidates : int = 0
    skipped_gap : int = 0
    skipped_eos : int = 0
    widened_band : int = 0
    kept_chosen : int = 0
    rows_out : int = 0
    negatives_per_chosen : int = 0
    mean_gap : float = 0.0
    median_gap : float = 0.0
    frac_rejected_more_likely : float = 0.0
    mean_eos_gap : float = 0.0
    near_miss : Dict[str, int] = field(default_factory = dict)
    top_replacements : List[Tuple[str, int]] = field(default_factory = list)
    seconds : float = 0.0


def build_split(rows : Sequence[dict], pool : Pool, scorer : BaseScorer, args, split : str) -> Tuple[List[dict], BuildReport]:
    t0 = time.perf_counter()
    rep = BuildReport(split = split, rows_in = len(rows), pool_words = len(pool.words), negatives_per_chosen = args.negatives)
    todo = rows[: args.limit] if args.limit else rows

    work = []
    for r in todo:
        rep.processed += 1
        lines = couplet_lines(r["chosen"])
        if len(lines) != 2:
            rep.skipped_not_two_lines += 1
            continue
        chosen = lines[0] + "\n" + lines[1]
        if not couplet_rhymes(chosen):
            rep.skipped_chosen_not_rhyme += 1
            continue
        w1, w2 = last_word(lines[0]), last_word(lines[1])
        k1 = rhyme_key(w1)
        cands = candidates(pool, w1, k1, w2, args.band, args.near_miss)
        if len(cands) < args.negatives:
            rep.widened_band += 1
            cands = candidates(pool, w1, k1, w2, 2.0 * args.band, args.near_miss)
        if len(cands) < args.negatives:
            rep.skipped_no_candidates += 1
            continue
        work.append((r, lines, chosen, w1, w2, cands))

    prefixes, cand_ids = [], []
    for r, lines, chosen, w1, w2, cands in work:
        prefix, _ = split_last(lines[1])
        prefixes.append(lines[0] + "\n" + prefix)
        cand_ids.append([scorer.first_id(w, after_space = bool(prefix)) for w in cands])
    slot = scorer.slot_logprobs([w[0]["prompt"] for w in work], prefixes, cand_ids) if work else []

    shortlist = []
    for (r, lines, chosen, w1, w2, cands), lp_slot in zip(work, slot):
        order = sorted(range(len(cands)), key = lambda j : (-lp_slot[j], cands[j]))
        shortlist.append([cands[j] for j in order[: args.candidates]])

    prompts, texts = [], []
    for (r, lines, chosen, w1, w2, cands), picked in zip(work, shortlist):
        prompts.append(r["prompt"])
        texts.append(chosen)
        for w in picked:
            prompts.append(r["prompt"])
            texts.append(replace_end(lines, w))
    lp, lp_end = scorer.logps(prompts, texts) if texts else ([], [])

    out, gaps, eos_gaps = [], [], []
    nm : collections.Counter = collections.Counter()
    reps : collections.Counter = collections.Counter()
    pos = 0
    for (r, lines, chosen, w1, w2, cands), picked in zip(work, shortlist):
        lp_c, end_c = lp[pos], lp_end[pos]
        lp_r = lp[pos + 1 : pos + 1 + len(picked)]
        end_r = lp_end[pos + 1 : pos + 1 + len(picked)]
        pos += 1 + len(picked)
        order = sorted(range(len(picked)), key = lambda j : (-lp_r[j], picked[j]))
        closes = [j for j in order if not scorer.append_eos or end_r[j] >= end_c - args.eos_gap]
        if len(closes) < args.negatives:
            rep.skipped_eos += 1
            continue
        keep = [j for j in closes if lp_r[j] >= lp_c - args.max_gap][: args.negatives]
        if len(keep) < args.negatives:
            rep.skipped_gap += 1
            continue
        rep.kept_chosen += 1
        for j in keep:
            w = picked[j]
            rejected = replace_end(lines, w)
            assert couplet_rhymes(chosen) and not couplet_rhymes(rejected), (chosen, rejected)
            b = near_miss_bucket(common_suffix(w1, w))
            out.append({
                "prompt" : r["prompt"],
                "chosen" : chosen,
                "rejected" : rejected,
                "domain" : r.get("domain", ""),
                "len_bucket" : r.get("len_bucket", ""),
                "near_miss" : b,
                "chosen_ends" : f"{w1}|{w2}",
                "rejected_ends" : f"{w1}|{w}",
                "subtheme" : r.get("subtheme", ""),
                "batch" : args.batch_tag,
                "base_logp_chosen" : round(lp_c, 4),
                "base_logp_rejected" : round(lp_r[j], 4),
                "base_eos_logp_chosen" : round(end_c, 4),
                "base_eos_logp_rejected" : round(end_r[j], 4),
            })
            gaps.append(lp_r[j] - lp_c)
            eos_gaps.append(end_c - end_r[j])
            nm[b] += 1
            reps[w] += 1

    rep.rows_out = len(out)
    if gaps:
        s = sorted(gaps)
        rep.mean_gap = sum(gaps) / len(gaps)
        rep.median_gap = s[len(s) // 2]
        rep.frac_rejected_more_likely = sum(1 for g in gaps if g > 0) / len(gaps)
        rep.mean_eos_gap = sum(eos_gaps) / len(eos_gaps)
    rep.near_miss = dict(sorted(nm.items()))
    rep.top_replacements = reps.most_common(15)
    rep.seconds = time.perf_counter() - t0
    return out, rep


def read_jsonl(path : Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path : Path, rows : Sequence[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")


def verify_loadable(path : Path, expected : int, slice_keys : Sequence[str]) -> int:
    res = load_groups(source = str(path), format = "pairwise", slice_keys = slice_keys)
    pairs = groups_to_pairs(res.groups)
    if len(pairs) != expected:
        raise RuntimeError(f"{path}: loader produced {len(pairs)} pairs for {expected} rows")
    return len(pairs)


def render(rep : BuildReport, width : int = 76) -> str:
    bar = "=" * width
    L = [bar, f"MINIMAL PAIRS  {rep.split}", bar,
         f"  rows in        : {rep.rows_in:,}   processed {rep.processed:,}   pool {rep.pool_words:,} end words   {rep.seconds:.1f}s",
         f"  skipped        : not two lines {rep.skipped_not_two_lines}   chosen not rhyme {rep.skipped_chosen_not_rhyme}"
         f"   no candidates {rep.skipped_no_candidates}   eos {rep.skipped_eos}   gap {rep.skipped_gap}   band widened {rep.widened_band}",
         f"  kept chosen    : {rep.kept_chosen:,}   x {rep.negatives_per_chosen} negatives = {rep.rows_out:,} rows",
         f"  logp gap       : mean {rep.mean_gap:+.3f}   median {rep.median_gap:+.3f} nats   rejected more likely {rep.frac_rejected_more_likely:.1%}"
         f"   eos gap mean {rep.mean_eos_gap:+.3f}",
         f"  near_miss      : {rep.near_miss}",
         "  replacements   : " + "  ".join(f"{w}:{c}" for w, c in rep.top_replacements),
         bar]
    return "\n".join(L)


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = "build minimal pair negatives from the chosen couplets of a pairwise dataset")
    ap.add_argument("--src", default = "data/preference_data/rhyme_50k", help = "folder holding <split>.jsonl")
    ap.add_argument("--out", default = "experiments/dpo/minimal_pairs/data", help = "folder that receives <split>.jsonl and report.json")
    ap.add_argument("--splits", default = "train,val")
    ap.add_argument("--negatives", type = int, default = 2, help = "rejected couplets per chosen")
    ap.add_argument("--candidates", type = int, default = 16, help = "slot ranked candidates scored as full sequences per chosen")
    ap.add_argument("--band", type = float, default = 1.5, help = "frequency band in log2 count around the replaced word")
    ap.add_argument("--max-gap", dest = "max_gap", type = float, default = 3.0,
                    help = "drop the couplet when fewer than --negatives candidates sit within this many nats below the chosen")
    ap.add_argument("--eos-gap", dest = "eos_gap", type = float, default = 1.5,
                    help = "drop a candidate whose EOS log prob after the new word sits more than this many nats below the chosen's")
    ap.add_argument("--near-miss", dest = "near_miss", choices = ("grade", "exclude"), default = "grade",
                    help = "grade keeps eye rhymes and labels them, exclude drops words sharing 2+ final letters")
    ap.add_argument("--model", default = "gpt2", help = "base model that measures plausibility, the DPO reference")
    ap.add_argument("--template", default = "{prompt}\n{response}", help = "must equal policy.template of the DPO config")
    ap.add_argument("--no-eos", dest = "append_eos", action = "store_false", help = "match policy.append_eos false")
    ap.add_argument("--device", default = "cpu")
    ap.add_argument("--batch-size", dest = "batch_size", type = int, default = 64, help = "sequences per forward")
    ap.add_argument("--limit", type = int, default = 0, help = "rows per split to process, 0 means all")
    ap.add_argument("--batch-tag", dest = "batch_tag", default = "minimal", help = "value written to the batch field")
    args = ap.parse_args(argv)

    if args.negatives < 1:
        raise SystemExit("--negatives must be >= 1")
    if args.candidates < args.negatives:
        raise SystemExit("--candidates must be >= --negatives")
    if args.band <= 0.0:
        raise SystemExit("--band must be > 0")
    if args.max_gap < 0.0:
        raise SystemExit("--max-gap must be >= 0")
    if args.eos_gap < 0.0:
        raise SystemExit("eos gap must be at least 0")
    if "{prompt}" not in args.template or "{response}" not in args.template:
        raise SystemExit("--template must contain {prompt} and {response}")

    src, out = Path(args.src), Path(args.out)
    slice_keys = ("domain", "len_bucket", "near_miss", "batch")
    scorer = BaseScorer(args.model, args.template, args.append_eos, args.device, args.batch_size)

    reports = {}
    for split in [s for s in args.splits.split(",") if s]:
        rows = read_jsonl(src / f"{split}.jsonl")
        pool = build_pool(rows)
        out_rows, rep = build_split(rows, pool, scorer, args, split)
        write_jsonl(out / f"{split}.jsonl", out_rows)
        verify_loadable(out / f"{split}.jsonl", len(out_rows), slice_keys)
        reports[split] = asdict(rep)
        print(render(rep))

    with open(out / "report.json", "w") as f:
        json.dump({"settings" : vars(args), "splits" : reports}, f, indent = 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
