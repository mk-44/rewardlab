from __future__ import annotations
import math
import re
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Literal
from rlhf.dataset.schema import PairView, PreferenceGroup, groups_to_pairs

_TOKEN_RE = re.compile(r"[a-z0-9']+|\n")
_PUNCT = set(".,;:!?\"'()[]{}-—")

def tokenize(text : str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

def pairwise_accuracy(pairs : Sequence[PairView], score : Callable[[str], float]) -> float:
    if not pairs:
        return 0.0

    won = tied = 0
    
    for p in pairs:
        sc, sr = score(p.chosen), score(p.rejected)
        if sc > sr:
            won += 1
        elif sc == sr:
            tied += 1
    
    return (won + 0.5 * tied) / len(pairs)

def _punct_count(t: str) -> float:
    return sum(1 for c in t if c in _PUNCT)

def _unique_word_ratio(t: str) -> float:
    w = t.lower().split()
    return len(set(w)) / len(w) if w else 0.0

TRIVIAL_BASELINES: dict[str, Callable[[str], float]] = {
    "char_length" : len,
    "word_count" : lambda t : len(t.split()),
    "newline_count" : lambda t : t.count("\n"),
    "punctuation" : _punct_count,
    "unique_word_ratio" : _unique_word_ratio,
    "avg_word_length" : lambda t : (sum(len(w) for w in t.split()) / len(t.split()) if t.split() else 0.0)
}


ProbeStatus = Literal["ok", "approximate", "unavailable"]

@dataclass
class LexicalProbe:
    accuracy : Optional[float] = None
    status : ProbeStatus = "unavailable"
    reason : str = "not run"
    n_train_pairs : int = 0
    n_test_pairs : int = 0
    vocab_size : int = 0
    top_chosen : list[tuple[str, float]] = field(default_factory=list)
    top_rejected : list[tuple[str, float]] = field(default_factory=list)
    note : str = "Naive Bayes is a weak model so this is a LOWER BOUND. A high value is decisive. A low value proves nothing."

    @property
    def usable(self) -> bool:
        return self.accuracy is not None and self.status != "unavailable"

def fit_lexical_probe(
    pairs : Sequence[PairView],
    test_frac : float = 0.2,
    min_count : int = 2,
    alpha : float = 1.0,
    top_k : int = 20,
    seed : int = 0,
    min_pairs : int = 50,
    min_test_for_confidence : int = 100,
) -> LexicalProbe:

    if len(pairs) < min_pairs:
        return LexicalProbe(reason = f"only {len(pairs)} pairs, need at least {min_pairs}")

    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    train = [pairs[i] for i in idx[: cut]]
    test = [pairs[i] for i in idx[cut: ]]

    cnt_c: Counter = Counter()
    cnt_r: Counter = Counter()
    for p in train:
        cnt_c.update(tokenize(p.chosen))
        cnt_r.update(tokenize(p.rejected))

    vocab = {t for t, n in (cnt_c + cnt_r).items() if n >= min_count}
    
    if not vocab:
        return LexicalProbe(
            reason = f"no token appeared at least {min_count} times",
            n_train_pairs = len(train),
            n_test_pairs = len(test)
        )
    
    if not test:
        return LexicalProbe(
            reason = f"test split is empty at test_frac={test_frac}",
            n_train_pairs = len(train)
        )
    
    tot_c = sum(cnt_c[t] for t in vocab)
    tot_r = sum(cnt_r[t] for t in vocab)
    V = len(vocab)
    denom_c = tot_c + alpha * V
    denom_r = tot_r + alpha * V

    log_odds = {
        t : math.log((cnt_c[t] + alpha) / denom_c) - math.log((cnt_r[t] + alpha) / denom_r)
        for t in vocab
    }

    def score(text : str) -> float:
        return sum(log_odds.get(t, 0.0) for t in tokenize(text))
    
    ranked = sorted(log_odds.items(), key = lambda kv: kv[1])
    thin = len(test) < min_test_for_confidence
    return LexicalProbe(
        accuracy = pairwise_accuracy(test, score),
        status = "approximate" if thin else "ok",
        reason = f"only {len(test)} test pairs, expect noise of a few points" if thin else "",
        n_train_pairs = len(train),
        n_test_pairs = len(test),
        vocab_size = V,
        top_chosen = [(t, round(v, 2)) for t, v in ranked[-top_k:][::-1]],
        top_rejected = [(t, round(v, 2)) for t, v in ranked[:top_k]],
    )


# surface_probes
_HEDGE = re.compile(r"\b(i think|it'?s important to note|generally|typically|in general|keep in mind|as always)\b", re.I)
_REFUSAL = re.compile(r"\b(i cannot|i can'?t|as an ai|i'?m unable|i apolog)", re.I)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")

SURFACE_PROBES : dict[str, Callable[[str], bool]] = {
    "has_newline" : lambda t : "\n" in t,
    "multi_line" : lambda t : t.count("\n") >= 1,
    "markdown_bold" : lambda t: "**" in t,
    "markdown_header" : lambda t: bool(re.search(r"^#{1,6} ", t, re.M)),
    "bullet_list" : lambda t: bool(re.search(r"^\s*[-*+] ", t, re.M)),
    "numbered_list" : lambda t: bool(re.search(r"^\s*\d+[.)] ", t, re.M)),
    "code_fence" : lambda t: "```" in t,
    "emoji" : lambda t: bool(_EMOJI.search(t)),
    "question_mark" : lambda t: "?" in t,
    "exclamation" : lambda t: "!" in t,
    "hedging" : lambda t: bool(_HEDGE.search(t)),
    "refusal" : lambda t: bool(_REFUSAL.search(t)),
    "ends_with_period" : lambda t: t.rstrip().endswith("."),
    "has_comma" : lambda t: "," in t,
    "repetitive" : lambda t: _unique_word_ratio(t) < 0.5,
}

@dataclass
class SurfaceProbe:
    name : str
    p_chosen : float
    p_rejected : float
    gap : float
    chi2 : float

def _chi2_2x2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return 0.0 if den == 0 else n * (a * d - b * c) ** 2 / den

def run_surface_probes(pairs : Sequence[PairView], probes : Optional[dict[str, Callable[[str], bool]]] = None) -> list[SurfaceProbe]:
    probes = probes or SURFACE_PROBES
    out = []
    n = len(pairs) or 1
    for name, fn in probes.items():
        a = sum(1 for p in pairs if fn(p.chosen))
        b = len(pairs) - a
        c = sum(1 for p in pairs if fn(p.rejected))
        d = len(pairs) - c
        out.append(
            SurfaceProbe(
                name = name,
                p_chosen = a / n,
                p_rejected = c / n,
                gap = (a - c) / n,
                chi2 = _chi2_2x2(a, b, c, d)
            )
        )
    return sorted(out, key = lambda s : -abs(s.gap))

@dataclass
class StructureStats:
    n_groups : int = 0
    n_pairs : int = 0
    n_unique_prompts : int = 0
    k_histogram : dict[int, int] = field(default_factory=dict)
    n_exact_duplicates : int = 0
    n_identical_sides : int = 0
    n_all_tied : int = 0
    n_empty_response : int = 0
    slice_keys : dict[str, int] = field(default_factory=dict)
    leakage_prompts : int = 0
    leakage_rate : float = 0.0

def structure_stats(groups : Sequence[PreferenceGroup], val_groups : Optional[Sequence[PreferenceGroup]] = None) -> StructureStats:
    st = StructureStats(n_groups = len(groups))
    seen : set[int] = set()
    prompts : set[str] = set()
    slice_vals : dict[str, set] = {}

    for g in groups:
        st.n_pairs += g.n_pairs()
        st.k_histogram[g.K] = st.k_histogram.get(g.K, 0) + 1
        prompts.add(g.prompt)

        key = hash((g.prompt, tuple(g.responses)))
        if key in seen:
            st.n_exact_duplicates += 1
        seen.add(key)

        if len(set(g.responses)) < len(g.responses):
            st.n_identical_sides += 1
        if not g.has_preference_signal:
            st.n_all_tied += 1
        if any(not r.strip() for r in g.responses):
            st.n_empty_response += 1

        for k, v in g.slices.items():
            slice_vals.setdefault(k, set()).add(v)

    st.n_unique_prompts = len(prompts)
    st.slice_keys = {k: len(v) for k, v in sorted(slice_vals.items())}

    if val_groups:
        val_prompts = {g.prompt for g in val_groups}
        overlap = prompts & val_prompts
        st.leakage_prompts = len(overlap)
        st.leakage_rate = len(overlap) / len(val_prompts) if val_prompts else 0.0

    return st

def _quantiles(xs : list[float], qs = (0.05, 0.5, 0.95)) -> dict[str, float]:
    if not xs:
        return {f"q{int(q*100):02d}": 0.0 for q in qs}
    s = sorted(xs)
    return {f"q{int(q*100):02d}": s[min(int(q * len(s)), len(s) - 1)] for q in qs}

@dataclass
class LengthStats:
    chosen : dict[str, float] = field(default_factory = lambda : _quantiles([]))
    rejected : dict[str, float] = field(default_factory = lambda : _quantiles([]))
    p_longer_wins : float = 0.0
    frac_near_equal : float = 0.0

def length_stats(pairs : Sequence[PairView], near_equal : float = 0.1) -> LengthStats:
    lc = [float(len(p.chosen)) for p in pairs]
    lr = [float(len(p.rejected)) for p in pairs]
    n = len(pairs) or 1
    longer = sum(1 for a, b in zip(lc, lr) if a > b)
    near = sum(1 for a, b in zip(lc, lr) if abs(a - b) / max(a, b, 1.0) < near_equal)
    return LengthStats(
        chosen = _quantiles(lc),
        rejected = _quantiles(lr),
        p_longer_wins = longer / n,
        frac_near_equal = near / n
    )

@dataclass
class AuditReport:
    source : str = ""
    expected_ceiling : float = 0.75
    structure : StructureStats = field(default_factory = StructureStats)
    lengths : LengthStats = field(default_factory = LengthStats)
    baselines : dict[str, float] = field(default_factory = dict)
    lexical : LexicalProbe = field(default_factory = LexicalProbe)
    surface : list[SurfaceProbe] = field(default_factory = list)
    warnings : list[str] = field(default_factory = list)

    @property
    def best_baseline(self) -> tuple[str, float, str]:
        if not self.baselines:
            return ("none", 0.0, "normal")
        name = max(self.baselines, key = lambda k: abs(self.baselines[k] - 0.5))
        acc = self.baselines[name]
        return (name, max(acc, 1 - acc), "flipped" if acc < 0.5 else "normal")

    def to_dict(self) -> dict:
        name, score, direction = self.best_baseline
        return {
            "source" : self.source,
            "expected_ceiling" : self.expected_ceiling,
            "structure" : self.structure.__dict__,
            "lengths" : self.lengths.__dict__,
            "lexical" : self.lexical.__dict__,
            "baselines" : self.baselines,
            "surface" : [s.__dict__ for s in self.surface],
            "baseline_acc_best" : score,
            "baseline_acc_best_name" : name,
            "baseline_acc_best_direction" : direction,
            "lexical_sep_acc" : self.lexical.accuracy,
            "lexical_sep_status" : self.lexical.status,
            "warnings" : self.warnings
        }

def build_warnings(rep: AuditReport, expected_ceiling: float = 0.75) -> list[str]:
    w = []
    st, lx = rep.structure, rep.lexical
    name, best, direction = rep.best_baseline
    tag = " [flipped]" if direction == "flipped" else ""

    if best > expected_ceiling:
        w.append(
            f"REGIME MISMATCH  a trivial baseline ({name}={best:.3f}{tag}) already exceeds expected_ceiling={expected_ceiling:.2f}."
            f" Human annotated data is not this separable, so either the declared regime is wrong or a shortcut is doing the work."
        )

    if lx.usable and lx.accuracy > 0.90:
        w.append(f"TASK IS LEXICALLY TRIVIAL  lexical_sep_acc={lx.accuracy:.3f} > 0.90. Bag of words alone nearly solves it.")
    
    if best > 0.80:
        w.append(f"A TRIVIAL MODEL ALREADY WINS  {name}={best:.3f}{tag} > 0.80. Any reward model must clearly beat this to mean anything.")

    if not 0.40 <= rep.lengths.p_longer_wins <= 0.60:
        w.append(f"LENGTH IMBALANCE  p_longer_wins={rep.lengths.p_longer_wins:.3f}. Chosen and rejected differ systematically in length.")

    if rep.surface and abs(rep.surface[0].gap) > 0.30:
        s = rep.surface[0]
        w.append(f"STRONG SURFACE SHORTCUT  {s.name} gap={s.gap:+.3f} (chosen {s.p_chosen:.2f} vs rejected {s.p_rejected:.2f}).")

    if st.leakage_prompts:
        w.append(f"TRAIN VAL LEAKAGE  {st.leakage_prompts} prompts shared ({st.leakage_rate:.1%} of val).")

    if st.n_identical_sides:
        w.append(f"BROKEN PAIRS  {st.n_identical_sides} groups where two responses are identical.")
    if st.n_all_tied:
        w.append(f"NO SIGNAL  {st.n_all_tied} groups are fully tied and yield zero pairs.")
    if st.n_empty_response:
        w.append(f"EMPTY RESPONSES  {st.n_empty_response} groups contain a blank response.")
    if not lx.usable:
        w.append(f"LEXICAL PROBE DID NOT RUN  {lx.reason}. Separability unknown.")
    if not st.slice_keys:
        w.append("NO SLICE KEYS  per slice metrics will report unavailable. Save a category field per row to enable them.")

    return w

def audit(
    groups : Sequence[PreferenceGroup],
    val_groups : Optional[Sequence[PreferenceGroup]] = None,
    source : str = "",
    seed : int = 0,
    expected_ceiling : float = 0.75,
) -> AuditReport:
    pairs = groups_to_pairs(groups)

    rep = AuditReport(source = source, expected_ceiling = expected_ceiling)
    rep.structure = structure_stats(groups, val_groups)
    rep.lengths = length_stats(pairs)
    rep.baselines = {n : pairwise_accuracy(pairs, fn) for n, fn in TRIVIAL_BASELINES.items()}
    rep.lexical = fit_lexical_probe(pairs, seed = seed)
    if rep.lexical.usable:
        rep.baselines["lexical_nb"] = rep.lexical.accuracy
    rep.surface = run_surface_probes(pairs)
    rep.warnings = build_warnings(rep, expected_ceiling)
    return rep

def render(rep: AuditReport, width: int = 76) -> str:
    st, ln, lx = rep.structure, rep.lengths, rep.lexical
    name, best, direction = rep.best_baseline
    L = []
    bar = "=" * width
    sub = lambda t: f"\n{'-'*4} {t} {'-'*(width-len(t)-6)}"

    L += [bar, "DATASET AUDIT   (tier 0: no model, no embeddings)", bar]
    L += [
        f"  source          : {rep.source or 'n/a'}",
        f"  groups          : {st.n_groups:,}      pairs : {st.n_pairs:,}",
        f"  unique prompts  : {st.n_unique_prompts:,}"
        f"      K : {dict(sorted(st.k_histogram.items()))}",
    ]

    L += [sub("STRUCTURE")]
    for label, v in [
        ("exact duplicates", st.n_exact_duplicates),
        ("identical sides", st.n_identical_sides),
        ("all tied (no signal)", st.n_all_tied),
        ("empty responses", st.n_empty_response),
        ("train val leakage", st.leakage_prompts),
    ]:
        flag = "" if v == 0 else "   <-- look"
        L.append(f"  {label:<24s}{v:>8,}{flag}")
    L.append(f"  {'slice keys':<24s}{str(st.slice_keys or 'none'):>8s}")

    L += [sub("HOW EASY IS THIS")]
    L.append("  a model that only knows ...            pairwise acc")
    for k, v in sorted(rep.baselines.items(), key=lambda kv: -kv[1]):
        mark = "  <-- best" if k == name else ""
        L.append(f"    {k:<34s}   {v:.3f}{mark}")
    L += [
        "",
        f"  >> baseline_acc_best = {best:.3f}  ({name}, {direction})"
        f"     [expected_ceiling {rep.expected_ceiling:.2f}]",
        f"     A reward model scoring below this has learned nothing.",
        "",
        f"  lexical probe [{lx.status}]: {lx.n_train_pairs:,} train / "
        f"{lx.n_test_pairs:,} test pairs, vocab {lx.vocab_size:,}",
    ]
    L.append(f"  {lx.note}" if lx.usable else f"  reason: {lx.reason}")

    if lx.top_chosen:
        L += [sub("WHAT GIVES IT AWAY")]
        fmt = lambda xs: "  ".join(f"{t!r}" for t, _ in xs[:12])
        L += [
            f"  predicts CHOSEN   : {fmt(lx.top_chosen)}",
            f"  predicts REJECTED : {fmt(lx.top_rejected)}",
        ]

    L += [sub("LENGTH")]
    L += [
        f"  chosen   chars  q05 {ln.chosen['q05']:.0f}   median {ln.chosen['q50']:.0f}"
        f"   q95 {ln.chosen['q95']:.0f}",
        f"  rejected chars  q05 {ln.rejected['q05']:.0f}   median {ln.rejected['q50']:.0f}"
        f"   q95 {ln.rejected['q95']:.0f}",
        f"  P(chosen is longer)   {ln.p_longer_wins:.3f}",
        f"  near equal length     {ln.frac_near_equal:.3f}"
        f"   <- the length controlled eval slice",
    ]

    L += [sub("SURFACE ASYMMETRY   (top 8 by gap)")]
    L.append(f"  {'feature':<20s}{'chosen':>9s}{'rejected':>10s}{'gap':>9s}")
    for s in rep.surface[:8]:
        L.append(
            f"  {s.name:<20s}{s.p_chosen:>9.3f}{s.p_rejected:>10.3f}{s.gap:>+9.3f}"
        )

    L += [sub("WARNINGS")]
    L += [f"  !  {w}" for w in rep.warnings] or ["  none"]
    L += ["", bar]
    return "\n".join(L)


