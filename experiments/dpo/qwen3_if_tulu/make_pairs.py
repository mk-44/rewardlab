import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from rlhf.core.preference.splits import render, resplit_files
from rlhf.dpo.collate import DPOCollator
from rlhf.dpo.config import load_dpo

DATASET = "allenai/tulu-3-pref-personas-instruction-following"
CONFIG = "experiments/dpo/qwen3_if_tulu/config.yaml"
VAL_FRAC = 0.025

cfg = load_dpo(CONFIG)
coll = DPOCollator(cfg.tokenizer_name(), cfg.policy.max_length, cfg.policy.template, cfg.policy.append_eos)
train_out, val_out = Path(cfg.data.train_path), Path(cfg.data.val_path)
OUT = train_out.parent / "all.jsonl"

rows = load_dataset(DATASET)["train"]
OUT.parent.mkdir(parents = True, exist_ok = True)

kept, identical, over_length = 0, 0, 0
families, n_constraints = Counter(), Counter()
words = {"prompt" : [], "chosen" : [], "rejected" : []}

with open(OUT, "w", encoding = "utf-8") as f:
    for row in rows:
        for side in ("chosen", "rejected"):
            assert [m["role"] for m in row[side]] == ["user", "assistant"], (row["id"], side)
            assert row[side][0]["content"] == row["prompt"], (row["id"], side)
        chosen, rejected = row["chosen"][1]["content"], row["rejected"][1]["content"]
        if chosen.strip() == rejected.strip():
            identical += 1
            continue
        if any(flag for response in (chosen, rejected) for flag in coll.encode_one(row["prompt"], response)[2:]):
            over_length += 1
            continue
        constraints = [c.strip() for c in row["constraints"]]
        rec = {
            "prompt" : row["prompt"],
            "chosen" : chosen,
            "rejected" : rejected,
            "id" : row["id"],
            "n_constraints" : len(constraints),
            "constraint_family" : constraints[0].split(":", 1)[0].strip(),
            "constraints" : " | ".join(constraints),
            "chosen_model" : row["chonsen_model"],
            "rejected_model" : row["rejected_model"],
        }
        f.write(json.dumps(rec, ensure_ascii = False) + "\n")
        kept += 1
        families[rec["constraint_family"]] += 1
        n_constraints[rec["n_constraints"]] += 1
        for k in words:
            words[k].append(len(rec[k].split()))

print(f"rows read {len(rows)}   kept {kept}   dropped as identical {identical}   dropped over max_length {cfg.policy.max_length} under the config template {over_length}   -> {OUT}")
print("n_constraints per row")
for n, c in sorted(n_constraints.items()):
    print(f"  {n}   {c}")
print("constraint_family of the first listed constraint")
for fam, c in families.most_common():
    print(f"  {fam:28s} {c}")
print("words per field   p50 / p90 / p99 / max")
for k, v in words.items():
    v = sorted(v)
    p = [v[round(q * (len(v) - 1))] for q in (0.5, 0.9, 0.99, 1.0)]
    print(f"  {k:10s} {p[0]:6d} {p[1]:6d} {p[2]:6d} {p[3]:6d}")

rep = resplit_files(
    inputs = [OUT],
    train_out = train_out,
    val_out = val_out,
    val_frac = VAL_FRAC,
    method = "hash",
)
print(render(rep))
(OUT.parent / "split_report.json").write_text(json.dumps(rep.to_dict(), indent = 2) + "\n")
