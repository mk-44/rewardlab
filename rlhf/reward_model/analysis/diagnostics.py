from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from rlhf.reward_model.core.contracts import ConfigError
import torch


def collect(model, batches : Sequence, device : str = "cpu"):
    was_train = model.training
    hc, hr, rc, rr, uids = [], [], [], [], []

    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                b = batch.to(device)
                chosen_hidden = model.backbone(b.chosen_input_ids, b.chosen_mask)
                rejected_hidden = model.backbone(b.rejected_input_ids, b.rejected_mask)
                chosen_reward = model.head(chosen_hidden)
                rejected_reward = model.head(rejected_hidden)

                rc.append(chosen_reward.detach().float().cpu())
                rr.append(rejected_reward.detach().float().cpu())
                hc.append(chosen_hidden.detach().float().cpu())
                hr.append(rejected_hidden.detach().float().cpu())
                uids.append(b.uids.detach().cpu())
    finally:
        model.train(was_train)
    
    if not hc:
        raise ConfigError("collect got zero batches.")
    
    return (torch.cat(hc), torch.cat(hr), torch.cat(rc), torch.cat(rr), torch.cat(uids))


@dataclass
class DirectionReport:
    n_pairs : int = 0
    accuracy : float = 0.0
    cos_w_mean_delta : float = 0.0
    cos_w_pc1 : float = 0.0
    pc1_explained_frac : float = 0.0
    delta_norm_q50 : float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _unit(x : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return x / x.norm(dim = dim, keepdim = True).clamp(min = 1e-12)


def pc_of1(A : torch.Tensor, n_iters : int = 64, seed : int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = _unit(torch.randn(A.shape[1], generator = g))

    for _ in range(n_iters):
        v = _unit(A.T @ (A @ v))
    return v


def directions(w : torch.Tensor, H_c : torch.Tensor, H_r : torch.Tensor) -> DirectionReport:
    diff = H_c - H_r
    margins = diff @ w.float()
    acc = float(((margins > 0).float() + (0.5) * (margins == 0).float()).mean())
    Du = _unit(diff)
    mean_dir = _unit(diff.mean(dim = 0), dim = 0)
    v = pc_of1(Du)

    if float(v @ mean_dir) < 0:
        v = -v
    
    proj = Du @ v
    w_unit = _unit(w.float(), dim = 0)

    return DirectionReport(
        n_pairs = int(diff.shape[0]),
        accuracy = acc,
        cos_w_mean_delta = float(w_unit @ mean_dir),
        cos_w_pc1 = float(w_unit @ v),
        pc1_explained_frac = float((proj ** 2).mean()),
        delta_norm_q50 = float(diff.norm(dim = 1).median())
    )


@dataclass
class SliceAcc:
    name : str = ""
    n_pairs : int = 0
    accuracy : float = 0.0
    accuracy_se : float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def slice_accuracy(pairs : Sequence, margins : torch.Tensor, min_pairs : int = 50):
    if len(pairs) != margins.shape[0]:
        raise ConfigError(f"num pairs : {len(pairs)} pairs should be equal to len_margins : {margins.shape[0]} margins")
    
    from rlhf.reward_model.dataset.distribution import build_slice_specs, slice_masks
    
    masks = slice_masks(pairs, build_slice_specs())
    correct = (margins > 0).float() + 0.5 * (margins == 0).float()
    out = []
    for name in sorted(masks):
        m = torch.as_tensor(masks[name], dtype = torch.bool)
        n = int(m.sum())
        if n < min_pairs:
            continue

        acc = float(correct[m].mean())
        out.append(SliceAcc(name = name, n_pairs = n, accuracy = acc, accuracy_se = float((acc * (1 - acc) / n) ** 0.5)))
    return out


@dataclass
class Extremes:
    worst : list = field(default_factory = list)
    best : list = field(default_factory = list)

    def to_dict(self) -> dict:
        return {"worst": [list(t) for t in self.worst], "best": [list(t) for t in self.best]}


def margin_extremes(uids: torch.Tensor, margins: torch.Tensor, k: int = 8) -> Extremes:
    order = torch.argsort(margins)
    take = min(k, margins.shape[0])
    return Extremes(
        worst = [(int(uids[i]), float(margins[i])) for i in order[: take]],
        best = [(int(uids[i]), float(margins[i])) for i in order.flip(0)[: take]]
    )


@dataclass
class CurveReport:
    n_steps_logged : int = 0
    loss_first : Optional[float] = None
    loss_last : Optional[float] = None
    eval_acc_series : list = field(default_factory=list)
    abs_reward_max_series : list = field(default_factory=list)
    mean_margin_first : Optional[float] = None
    mean_margin_last : Optional[float] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_curves(run_dir) -> CurveReport:
    from rlhf.reward_model.core.logging import read_metrics
    path = Path(run_dir) / "metrics.jsonl"

    if not path.exists():
        raise ConfigError(f"no metrics.jsonl in {run_dir}")
    
    rows = read_metrics(path)
    losses = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    accs = [(r["step"], r["eval/accuracy"]) for r in rows if "eval/accuracy" in r]
    rmax = [(r["step"], r["eval/abs_reward_max"]) for r in rows if "eval/abs_reward_max" in r]
    marg = [(r["step"], r["eval/mean_margin"]) for r in rows if "eval/mean_margin" in r]
    return CurveReport(
        n_steps_logged = len(losses),
        loss_first = losses[0][1] if losses else None,
        loss_last = losses[-1][1] if losses else None,
        eval_acc_series = accs,
        abs_reward_max_series = rmax,
        mean_margin_first = marg[0][1] if marg else None,
        mean_margin_last = marg[-1][1] if marg else None
    )


@dataclass
class Tier2Report:
    source : str = ""
    direction : DirectionReport = field(default_factory = DirectionReport)
    slices : list = field(default_factory = list)
    extremes : Extremes = field(default_factory = Extremes)
    curves : Optional[CurveReport] = None

    def to_dict(self) -> dict:
        return {
            "source" : self.source,
            "direction" : self.direction.to_dict(),
            "slices" : [s.to_dict() for s in self.slices],
            "extremes" : self.extremes.to_dict(),
            "curves" : self.curves.to_dict() if self.curves else None
        }


def diagnose(
    model, 
    batches, 
    pairs : Optional[Sequence] = None,
    run_dir = None, 
    device : str = "cpu", 
    k_extremes : int = 8,
    source : str = ""
) -> Tier2Report:
    H_c, H_r, r_c, r_r, uids = collect(model, batches, device)
    margins = r_c - r_r
    w = model.head.linear.weight[0].detach()
    rep = Tier2Report(source = source)
    rep.direction = directions(w, H_c, H_r)
    if pairs is not None:
        rep.slices = slice_accuracy(pairs, margins)
    rep.extremes = margin_extremes(uids, margins, k=k_extremes)
    if run_dir is not None:
        rep.curves = run_curves(run_dir)
    return rep


def render(rep: Tier2Report, width: int = 76) -> str:
    L = []
    bar = "=" * width
    L.append(bar)
    L.append("TIER 2 DIAGNOSTICS   (the trained model vs the predictions)")
    L.append(bar)
    if rep.source:
        L.append(f"  source          : {rep.source}")
    d = rep.direction
    L.append("")
    L.append("---- DIRECTION " + "-" * (width - 15))
    L.append(f"  val pairs         {d.n_pairs:,}     accuracy {d.accuracy:.3f}")
    L.append(f"  cos(w, mean delta)  {d.cos_w_mean_delta:+.3f}")
    L.append(f"  cos(w, PC1)         {d.cos_w_pc1:+.3f}   (PC1 explains {d.pc1_explained_frac:.1%} of unit-delta var)")
    L.append(f"  ||delta|| median    {d.delta_norm_q50:.3f}")
    if rep.slices:
        L.append("")
        L.append("---- SLICES (Tier-1 registry) " + "-" * (width - 30))
        L.append(f"  {'slice':28s} {'pairs':>7s} {'acc':>7s} {'se':>7s}")
        for s in sorted(rep.slices, key=lambda s: s.accuracy):
            L.append(f"  {s.name:28s} {s.n_pairs:>7,} {s.accuracy:>7.3f} {s.accuracy_se:>7.3f}")
    e = rep.extremes
    if e.worst:
        L.append("")
        L.append("---- EXTREMES (uid, margin) " + "-" * (width - 28))
        L.append("  worst: " + "  ".join(f"({u}, {m:+.2f})" for u, m in e.worst[:4]))
        L.append("  best : " + "  ".join(f"({u}, {m:+.2f})" for u, m in e.best[:4]))
    c = rep.curves
    if c is not None:
        L.append("")
        L.append("---- RUN CURVES " + "-" * (width - 16))
        L.append(f"  loss  {c.loss_first} -> {c.loss_last}   ({c.n_steps_logged} logged steps)")
        if c.eval_acc_series:
            L.append(f"  eval acc  " + " -> ".join(f"{a:.3f}" for _, a in c.eval_acc_series[-4:]))
        if c.abs_reward_max_series:
            first, last = c.abs_reward_max_series[0][1], c.abs_reward_max_series[-1][1]
            L.append(f"  |reward| max  {first:.2f} -> {last:.2f}   <- the inflation canary")
    L.append(bar)
    return "\n".join(L)
