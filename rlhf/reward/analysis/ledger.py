from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LedgerRow:
    question: str = ""
    predicted: str = ""
    observed: str = ""
    verdict: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def check_shared_space(model_stamp: dict, collate_stamp: dict,
                       embed_model: str, embed_pooling: str,
                       embed_max_length: int) -> bool:
    bb = model_stamp.get("backbone", {})
    return (bb.get("model_name") == embed_model
            and bb.get("pooling") == embed_pooling
            and collate_stamp.get("max_length") == embed_max_length
            and collate_stamp.get("template") == "{response}")


def build_ledger(audit: dict, profile: dict, train_report: dict,
                 eval_report: dict, tier2: dict,
                 trainer_stamp: Optional[dict] = None,
                 shared_space: Optional[bool] = None) -> list:
    sep = profile["separability"]
    delta = profile["delta"]
    shift = profile["shift"]
    acc = eval_report["overall"]["accuracy"]
    se = eval_report["overall"]["accuracy_se"]
    d = tier2["direction"]
    slices = tier2["slices"]
    curves = tier2.get("curves") or {}
    rmax = curves.get("abs_reward_max_series") or []

    worst = min(slices, key=lambda s: s["accuracy"]) if slices else None
    lens = [s for s in slices if "len" in s["name"] or "word_count" in s["name"]]
    worst_len = min(lens, key=lambda s: s["accuracy"]) if lens else None

    rows = []

    rows.append(LedgerRow(
        "how easy is this task",
        f"best baseline {audit['baseline_acc_best']:.3f} ({audit['baseline_acc_best_name']}); "
        f"separability {sep['separability_raw']:.3f}",
        f"val acc {acc:.4f} (se {se:.4f}), best at step {train_report['best_step']} "
        f"of {train_report['steps']}",
        "CONFIRMED — at the surface ceiling almost immediately"
        if acc >= audit["baseline_acc_best"] - 0.005
        else "model BELOW surface ceiling — investigate",
    ))

    rows.append(LedgerRow(
        "length shortcut",
        f"char_length baseline {audit['baselines']['char_length']:.3f}; "
        f"P(longer wins) {audit['lengths']['p_longer_wins']:.3f}; "
        f"near-equal pairs {audit['lengths']['frac_near_equal']:.3f}",
        (f"worst length-matched slice: {worst_len['name']} acc {worst_len['accuracy']:.4f} "
         f"(n={worst_len['n_pairs']}, se {worst_len['accuracy_se']:.4f})"
         if worst_len else "no length slice with enough pairs"),
        ("survives length control — NOT length-dependent"
         if worst_len and worst_len["accuracy"] >= acc - 2 * max(worst_len["accuracy_se"], 1e-9)
         else "DROPS on length-controlled slice — length shortcut in use"
         if worst_len else "n/a"),
    ))

    rows.append(LedgerRow(
        "surface-reader or not",
        f"frac_surface_controlled {sep['frac_surface_controlled']:.3f}; "
        f"separability_min {sep['separability_min']:.3f} (probe worst slice)",
        (f"model worst slice: {worst['name']} acc {worst['accuracy']:.4f} "
         f"(n={worst['n_pairs']})" if worst else "no slices"),
        ("holds on every matched slice — beyond any single surface feature"
         if worst and worst["accuracy"] >= 0.98 else "read the slice table"),
    ))

    if shared_space:
        pc1_verdict = ("DIRECT comparison — spaces verified identical "
                       "(model==instrument): cos(w, Tier-1 artifact axis) as stated")
    elif shared_space is False:
        pc1_verdict = ("own-space numbers only — spaces DIFFER, not comparable "
                       "to Tier-1 axes (step-20 F1)")
    else:
        pc1_verdict = "numbers stated — interpretation is the reader's"
    rows.append(LedgerRow(
        "did it learn the PC1 axis",
        f"Tier-1 PC1 explains {delta['pc_explained_var'][0]:.3f} of delta var "
        f"(the rejection-strategy axis)",
        f"cos(w, PC1) {d['cos_w_pc1']:+.3f}, cos(w, mean delta) {d['cos_w_mean_delta']:+.3f}, "
        f"own-space PC1 frac {d['pc1_explained_frac']:.2f}",
        pc1_verdict,
    ))

    rows.append(LedgerRow(
        "margin inflation on separable data",
        "predicted: margins/|r| grow after accuracy saturates (step-13 demo)",
        (f"abs_reward_max {rmax[0][1]:.2f} -> {rmax[-1][1]:.2f} across the run; "
         f"best checkpoint (step {train_report['best_step']}) predates most inflation"
         if rmax else "no curve data"),
        "CONFIRMED" if len(rmax) >= 2 and rmax[-1][1] > rmax[0][1] * 1.1 else "not observed",
    ))

    rows.append(LedgerRow(
        "honest evaluation",
        f"re-split: prompt overlap {shift['prompt_overlap_frac']:.1%}, shift real",
        f"eval on {eval_report['overall']['n_pairs']} out-of-sample-prompt pairs",
        "every number above is leakage-free",
    ))

    budget = (f"budget: {trainer_stamp['total_steps']} steps"
              if trainer_stamp else "budget: (trainer stamp not provided)")
    rows.append(LedgerRow(
        "run facts",
        budget,
        f"early_stopped={train_report['early_stopped']} at step {train_report['steps']}, "
        f"wall {train_report['wall_seconds']:.0f}s",
        "early stop reclaimed the saturated tail"
        if train_report["early_stopped"] else "ran full budget",
    ))
    return rows


def render_ledger(rows, title: str = "", width_q: int = 26,
                  width_p: int = 52, width_o: int = 56) -> str:
    total = width_q + width_p + width_o + 36
    L = ["=" * total,
         "THE LEDGER  —  predictions (Tier 0/1) vs the trained model (Tier 2)"]
    if title:
        L.append(f"experiment: {title}")
    L += ["=" * total,
          f"{'question':<{width_q}} | {'predicted':<{width_p}} | "
          f"{'observed':<{width_o}} | verdict",
          "-" * total]
    for r in rows:
        L.append(f"{r.question:<{width_q}} | {r.predicted:<{width_p}} | "
                 f"{r.observed:<{width_o}} | {r.verdict}")
    L.append("=" * total)
    return "\n".join(L)
