from __future__ import annotations
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, Optional, Callable, Iterable
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rlhf.reward_model.dataset.audit import SURFACE_PROBES, TRIVIAL_BASELINES
from rlhf.reward_model.dataset.schema import PairView, groups_to_pairs
from scipy.optimize import minimize

Status = Literal["ok", "approximate", "unavailable", "not_applicable", "degenerate", "error"]

@dataclass
class Block:
    status : Status = "unavailable"
    reason : str = "not run"

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "approximate")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, np.ndarray)}

def _fail(block : Block, status : Status, reason : str) -> Block:
    block.status, block.reason = status, reason
    return block

def unit_normalize(X : np.ndarray, eps : float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis = 1, keepdims = True)
    return X / np.maximum(n, eps)

def mean_pairwise_cosine(U : np.ndarray) -> float:
    n = len(U)
    if n < 2:
        return float("nan")
    m = U.sum(axis = 0)
    return float((m @ m - n) / (n * (n - 1)))

def spectrum(X : np.ndarray, normalize_rows : bool = True) -> np.ndarray:
    if normalize_rows:
        X = unit_normalize(X)
    n, d = X.shape
    if n == 0:
        return np.zeros(0)
    if n < d:
        lam = np.linalg.eigvalsh(X @ X.T / n)
    else:
        lam = np.linalg.eigvalsh(X.T @ X / n)
    
    return np.clip(lam, 0.0, None)[::-1]

def vendi_score(X : np.ndarray, normalize_rows : bool = True) -> float:
    lam = spectrum(X, normalize_rows)
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return 0.0
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))

def participation_ratio(lam : np.ndarray) -> float:
    num = float(lam.sum())
    den = float((lam * lam).sum())
    return float((num * num) / den) if den > 0.0 else 0.0

def top_directions(X : np.ndarray, k : int = 3, iters : int = 64, seed : int = 0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    total_var = float((X * X).sum()) / max(n, 1)
    dirs, eigs = [], []

    for _ in range(min(k, d)):
        v = rng.normal(size = d)
        v /= np.linalg.norm(v)
        for _ in range(iters):
            w = X @ v
            v = X.T @ w
            for u in dirs:
                v -= (v @ u) * u
            n_v = np.linalg.norm(v)
            if n_v < 1e-12:
                break
            v /= n_v
        lam = float(((X @ v) ** 2).sum()) / max(n, 1)
        dirs.append(v)
        eigs.append(lam)
    
    return np.array(dirs), np.array(eigs), total_var    

def top_k_indices(scores : np.ndarray, k : int, largest : bool = True) -> np.ndarray:
    k = min(k, len(scores))
    if k == 0:
        return np.zeros(0, dtype = int)
    s = -scores if largest else scores
    idx = np.argpartition(s, k - 1)[: k]
    return idx[np.argsort(s[idx])]

@dataclass
class EmbedConfig:
    backend : Literal["hf", "openai"] = "hf"
    model : str = "sentence-transformers/all-MiniLM-L6-v2"
    device : str = "cpu"
    embed_pooling_method : Literal["cls", "mean"] = "mean"
    batch_size : int = 64
    max_length : int = 512
    cache_dir : str = ".cache/embeddings"
    max_samples : int = 20_000
    seed : int = 0

def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

class EmbeddingCache:
    def __init__(self, cache_dir: str, model_id: str):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
        self.dir = Path(cache_dir) / safe
        self.dir.mkdir(parents=True, exist_ok=True)
        self.idx_path = self.dir / "index.json"
        self.vec_path = self.dir / "vectors.npy"
        self.index: dict[str, int] = {}
        self.vectors: Optional[np.ndarray] = None
        self.n_lookups = 0
        self.n_hits = 0
        self.n_embedded = 0
        self._load()

    def _load(self) -> None:
        if self.idx_path.exists() and self.vec_path.exists():
            self.index = json.loads(self.idx_path.read_text())
            self.vectors = np.load(self.vec_path)

    def get_many(self, texts: Sequence[str]):
        hits, misses, seen = {}, [], set()
        for t in texts:
            self.n_lookups += 1
            h = sha256_text(t)
            if h in self.index:
                hits[t] = self.index[h]
                self.n_hits += 1
            elif h not in seen:
                seen.add(h)
                misses.append(t)
        return hits, misses

    def put_many(self, texts: Sequence[str], vecs: np.ndarray) -> None:
        if len(texts) == 0:
            return
        base = 0 if self.vectors is None else len(self.vectors)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        for i, t in enumerate(texts):
            self.index[sha256_text(t)] = base + i
        self.n_embedded += len(texts)
        np.save(self.vec_path, self.vectors)
        tmp = self.idx_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index))
        os.replace(tmp, self.idx_path)

def _embed_hf(texts : Sequence[str], cfg : EmbedConfig) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(cfg.model)
    emb_model = AutoModel.from_pretrained(cfg.model).to(cfg.device).eval()
    
    out_embeds = []

    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size):
            txt = list(texts[i : i + cfg.batch_size])
            tok_out = tok(txt, return_tensors = "pt", max_length = cfg.max_length, padding = True, truncation = True).to(cfg.device)
            last_state = emb_model(**tok_out).last_hidden_state
            if cfg.embed_pooling_method == "mean":
                emb = (last_state * tok_out["attention_mask"].unsqueeze(-1)).sum(dim = 1) / (tok_out["attention_mask"].unsqueeze(-1).sum(dim = 1).clamp(min = 1e-9))
                out_embeds.append(emb.detach().float().cpu().numpy())
            else:
                emb = last_state[:, 0, :]
                out_embeds.append(emb.detach().float().cpu().numpy())

    out_embeds = np.vstack(out_embeds).astype(np.float32)
    return out_embeds

def _embed_openai(texts : Sequence[str], cfg : EmbedConfig) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai not installed. Pls run `pip install openai`. ") from e

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not in env")

    client = OpenAI(api_key=key)
    out = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = list(texts[i: i + cfg.batch_size])
        resp = client.embeddings.create(model=cfg.model, input=batch)
        out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
    return np.vstack(out)

_BACKENDS : dict[str, Callable[[Sequence[str], EmbedConfig], np.ndarray]] = {"hf": _embed_hf, "openai": _embed_openai}

def embed(texts : Sequence[str], cfg : EmbedConfig, cache: Optional[EmbeddingCache] = None):
    if cfg.backend not in _BACKENDS:
        raise RuntimeError(f"cfg :{cfg.backend} not available. Pls select among {list(_BACKENDS.keys())}")
    
    cache = cache or EmbeddingCache(cfg.cache_dir, f"{cfg.backend}:{cfg.model}")
    hits, misses = cache.get_many(texts)

    if misses:
        misses_vecs = _BACKENDS[cfg.backend](misses, cfg)
        if len(misses_vecs) != len(misses):
            raise RuntimeError(f"Embedded sent {len(misses_vecs)} vectors for {len(misses)} texts, which are not equal")
        cache.put_many(misses, misses_vecs)

        hits.update({t : cache.index[sha256_text(t)] for t in misses})
    
    rows = [hits[t] for t in texts]
    return cache.vectors[rows].astype(np.float32), cache

@dataclass
class EmbeddingHealth(Block):
    backend : str = ""
    model_id : str = ""
    dim : int = 0
    n_texts : int = 0
    n_unique : int = 0
    n_lookups : int = 0
    n_cache_hits : int = 0
    n_embedded : int = 0
    mean_norm : Optional[float] = None
    frac_zero_vectors : Optional[float] = None
    effective_rank : Optional[float] = None

def embedding_health(X : np.ndarray, cfg : EmbedConfig, cache : EmbeddingCache, n_texts : int, min_rank : float = 2.0) -> EmbeddingHealth:
    h = EmbeddingHealth(
        backend = cfg.backend, 
        model_id = cfg.model, 
        dim = int(X.shape[1]) if X.size else 0,
        n_texts = n_texts, 
        n_unique = len(cache.index),
        n_lookups = cache.n_lookups,
        n_cache_hits = cache.n_hits, 
        n_embedded = cache.n_embedded
    )

    if X.size == 0:
        return _fail(h, "unavailable", "No text embedded.")

    norms = np.linalg.norm(X, axis=1)
    h.mean_norm = float(norms.mean())
    h.frac_zero_vectors = float((norms < 1e-8).mean())
    h.effective_rank = participation_ratio(spectrum(X))

    if h.frac_zero_vectors > 0.01:
        return _fail(h, "degenerate", f"{h.frac_zero_vectors:.1%} vectors are zero")
    if h.effective_rank < min_rank:
        return _fail(h, "degenerate", f"effective_rank {h.effective_rank:.2f} < {min_rank}, collapsed.")
    return _fail(h, "ok", "")

@dataclass(frozen=True)
class SliceSpec:
    name : str
    kind : Literal["all", "bool_matched", "num_matched", "all_matched"]
    fn : Optional[Callable[[str], object]] = None


def build_slice_specs() -> list[SliceSpec]:
    specs = [SliceSpec("raw", "all")]
    specs += [SliceSpec(f"match_{n}", "bool_matched", f) for n, f in SURFACE_PROBES.items()]
    specs += [SliceSpec(f"match_{n}", "num_matched", f) for n, f in TRIVIAL_BASELINES.items()]
    specs += [SliceSpec("all_matched", "all_matched")]
    return specs

def _memo(texts : Iterable[str], fn : Callable[[str], object]) -> dict:
    out = {}
    for t in texts:
        if t not in out:
            out[t] = fn(t)
    return out

def slice_masks(pairs : Sequence[PairView], specs : Sequence[SliceSpec], eps : float = 0.1) -> dict:
    n = len(pairs)
    C, R = [p.chosen for p in pairs], [p.rejected for p in pairs]
    masks : dict[str, np.ndarray] = {}
    matched_all = np.ones(n, dtype = bool)

    for sp in specs:
        if sp.kind == "all":
            masks[sp.name] = np.ones(n, dtype=bool)
            continue
            
        if sp.kind == "all_matched":
            continue

        memo = _memo(C + R, sp.fn)
        vc = np.array([memo[t] for t in C])
        vr = np.array([memo[t] for t in R])

        if sp.kind == "bool_matched":
            m = (vc == vr)
        else:
            vc = vc.astype(np.float64)
            vr = vr.astype(np.float64)
            denom = np.maximum(np.maximum(np.abs(vc), np.abs(vr)), 1.0)
            m = (np.abs(vc - vr) / denom) < eps

        masks[sp.name] = m
        matched_all &= m

    if any(sp.kind == "all_matched" for sp in specs):
        masks["all_matched"] = matched_all
    return masks

def group_aware_split(pairs : Sequence[PairView], test_frac : float = 0.2, seed : int = 0):
    prompts = sorted({p.prompt for p in pairs})
    rng = np.random.default_rng(seed)
    keep = rng.random(len(prompts)) >= test_frac
    is_train = {pr : bool(k) for pr, k in zip(prompts, keep)}
    tr = np.array([is_train[p.prompt] for p in pairs], dtype = bool)
    return tr, ~tr

def _sigmoid(z : np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    neg = ~pos
    exp_z_neg = np.exp(z[neg])
    out[neg] = exp_z_neg / (1 + exp_z_neg)
    return out

def fit_linear_probe(D: np.ndarray, l2: float = 1.0, max_iter: int = 200):
    n, d = D.shape
    if n == 0:
        return np.zeros(d)

    def obj(w):
        z = D @ w
        loss = np.logaddexp(0.0, -z).sum() + l2 * (w @ w)
        g = -(D.T @ _sigmoid(-z)) + 2.0 * l2 * w
        return loss, g
    res = minimize(obj, np.zeros(d), jac = True, method = "L-BFGS-B", options = {"maxiter" : max_iter})
    return res.x

def pairwise_auc(s : np.ndarray) -> float:
    # Unserstand as prob(random_pos_score > random_neg_score) + 0.5 (equal_case_prob)
    n = len(s)
    if n == 0:
        return float("nan")
    sorted_s = np.sort(s)
    left = np.searchsorted(sorted_s, -s, side = "left") # vals < -s
    right = np.searchsorted(sorted_s, -s, side = "right") # vals <= -s
    gt = n - right
    eq = right - left

    return float((gt.sum() + (0.5 * eq.sum())) / (n * n))

@dataclass
class SliceResult(Block):
    name : str = ""
    n_pairs : int = 0
    n_train : int = 0
    n_test : int = 0
    acc : Optional[float] = None
    auc : Optional[float] = None
    drop_vs_raw : Optional[float] = None

@dataclass
class Separability(Block):
    slices : dict = field(default_factory = dict)
    frac_surface_controlled : Optional[float] = None
    separability_raw : Optional[float] = None
    separability_min : Optional[float] = None
    top_confound : Optional[str] = None
    top_confound_drop : Optional[float] = None

def separability(
    pairs : Sequence[PairView], 
    emb_chosen : np.ndarray, 
    emb_rejected : np.ndarray, 
    masks : dict,
    min_pairs : int = 100, 
    min_confident : int = 500,
    min_train_samples : int = 20,
    min_test_samples : int = 20,
    test_frac : float = 0.2, 
    seed : int = 0
) -> Separability:
    
    sep = Separability()
    emb_diff = emb_chosen - emb_rejected
    out : dict[str, SliceResult] = {}

    for name, m in masks.items():
        sl_res = SliceResult(name = name, n_pairs = int(m.sum()))
        if sl_res.n_pairs < min_pairs:
            out[name] = _fail(sl_res, status = "unavailable", reason = f"only {sl_res.n_pairs} pairs, need {min_pairs}")
            continue
            
        m_subset = [p for p, pair_matched in zip(pairs, m) if pair_matched == True]
        train_set, test_set = group_aware_split(m_subset, test_frac, seed)
        sl_res.n_train = int(train_set.sum())
        sl_res.n_test = int(test_set.sum())

        if sl_res.n_train < min_train_samples or sl_res.n_test < min_test_samples:
            out[name] = _fail(
                sl_res, 
                "unavailable", 
                f"split should have {min_train_samples} but has {sl_res.n_train}, {min_test_samples} but has {sl_res.n_test}"
            )
            continue
            
        emb_diff_sl = emb_diff[m]
        wts = fit_linear_probe(emb_diff_sl[train_set])
        sl_test_scores = emb_diff_sl[test_set] @ wts
        sl_res.acc = float(((sl_test_scores > 0) + 0.5 * (sl_test_scores == 0)).mean())
        sl_res.auc = pairwise_auc(sl_test_scores)

        thin = (sl_res.n_pairs < min_confident)
        out[name] = _fail(sl_res, "approximate" if thin else "ok", f"only {sl_res.n_pairs} pairs, expect noise" if thin else "")
    
    sep.slices = out
    if "all_matched" in out and len(pairs):
        sep.frac_surface_controlled = out["all_matched"].n_pairs / len(pairs)
    raw = out.get("raw")
    if raw is None or not raw.usable:
        return _fail(sep, "unavailable", "Raw slice did not run.")
    
    sep.separability_raw = raw.acc
    for name, sl in out.items():
        if sl.usable and name != "raw":
            sl.drop_vs_raw = raw.acc - sl.acc
    
    live = {n : sl for n, sl in out.items() if sl.usable}
    sep.separability_min = min(sl.acc for sl in live.values())
    controls = {n : sl for n, sl in live.items() if n != "raw"}
    if controls:
        sl_max = max(controls, key = lambda n : controls[n].drop_vs_raw)
        sep.top_confound = sl_max
        sep.top_confound_drop = controls[sl_max].drop_vs_raw
    return _fail(sep, "ok", "")

@dataclass
class DeltaDirection(Block):
    n_pairs : int = 0
    mean_pairwise_cosine : Optional[float] = None
    frac_aligned_with_mean : Optional[float] = None
    pc_explained_var : list = field(default_factory = list)
    effective_dim : Optional[float] = None
    effective_dim_unit : Optional[float] = None
    delta_norm_q05 : Optional[float] = None
    delta_norm_q50 : Optional[float] = None
    delta_norm_q95 : Optional[float] = None
    pc1_top : list = field(default_factory = list)
    pc1_bottom : list = field(default_factory = list)

def delta_direction(
    pairs : Sequence[PairView], 
    emb_chosen : np.ndarray, 
    emb_rejected : np.ndarray, 
    k : int = 3,
    n_examples : int = 5, 
    min_pairs : int = 50
) -> DeltaDirection:

    dd = DeltaDirection(n_pairs = len(pairs))
    if len(pairs) < min_pairs:
        return _fail(dd, "unavailable", f"only {len(pairs)} pairs, need {min_pairs}")
    diff = emb_chosen - emb_rejected
    norms = np.linalg.norm(diff, axis = 1)
    dd.delta_norm_q05, dd.delta_norm_q50, dd.delta_norm_q95 = (float(v) for v in np.quantile(norms, [0.05, 0.5, 0.95]))
    U = unit_normalize(diff)
    dd.mean_pairwise_cosine = mean_pairwise_cosine(U)
    
    mean_dir = U.mean(axis = 0)
    mean_norm = np.linalg.norm(mean_dir)
    if mean_norm < 1e-12:
        return _fail(dd, "degenerate", "mean delta direction is zero")
    
    dd.frac_aligned_with_mean = float(((U @ (mean_dir / mean_norm)) > 0).mean())
    dirs, eig_vals, total_var = top_directions(diff, k = k)
    
    if dirs[0] @ mean_dir < 0:
        dirs[0] = -dirs[0]
    
    dd.pc_explained_var = [float(e / total_var) if total_var > 0 else 0.0 for e in eig_vals]
    dd.effective_dim = participation_ratio(spectrum(diff, normalize_rows = False))
    dd.effective_dim_unit = participation_ratio(spectrum(U, normalize_rows = False))

    proj = diff @ dirs[0]
    hi = top_k_indices(proj, n_examples, largest = True)
    lo = top_k_indices(proj, n_examples, largest = False)
    dd.pc1_top = [(pairs[i].chosen, pairs[i].rejected) for i in hi]
    dd.pc1_bottom = [(pairs[i].chosen, pairs[i].rejected) for i in lo]
    return _fail(dd, "ok", "")

@dataclass
class Grounding(Block):
    cos_prompt_chosen : Optional[float] = None
    cos_prompt_rejected : Optional[float] = None
    relevance_gap : Optional[float] = None
    frac_rejected_offtopic : Optional[float] = None

def grounding(emb_prompt : np.ndarray, emb_chosen : np.ndarray, emb_rejected : np.ndarray, offtopic_tau : float = 0.2) -> Grounding:
    g = Grounding()
    if len(emb_prompt) == 0:
        return _fail(g, "unavailable", "No prompt embedding present.")

    P, C, R = unit_normalize(emb_prompt), unit_normalize(emb_chosen), unit_normalize(emb_rejected)
    sc = (P * C).sum(axis=1)
    sr = (P * R).sum(axis=1)
    g.cos_prompt_chosen = float(sc.mean())
    g.cos_prompt_rejected = float(sr.mean())
    g.relevance_gap = g.cos_prompt_chosen - g.cos_prompt_rejected
    g.frac_rejected_offtopic = float((sr < offtopic_tau).mean()) # Shortcut check
    return _fail(g, "ok", "")

def _rbf_kernel(Z : np.ndarray, gamma : Optional[float] = None):
    sq = (Z * Z).sum(axis = 1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - (2.0 * (Z @ Z.T)), 0.0)

    if gamma is None:
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1 / max(med, 1e-12)
    
    return np.exp(-gamma * d2), float(gamma)

def _mmd2_from_K(K : np.ndarray, Zind : np.ndarray, m : int, k : int) -> np.ndarray:
    KZ = K @ Zind
    S_xx = (Zind * KZ).sum(axis = 0)
    row = K.sum(axis = 1)
    zK1 = Zind.T @ row
    total = float(K.sum())
    S_xy = zK1 - S_xx
    S_yy = total - 2.0 * zK1 + S_xx
    dg = np.diag(K)
    dx = Zind.T @ dg
    dy = float(dg.sum()) - dx
    return (S_xx - dx) / (m * (m - 1)) + (S_yy - dy) / (k * (k - 1)) - 2.0 * S_xy / (m * k)

@dataclass
class ShiftResult(Block):
    name : str = ""
    mmd2 : Optional[float] = None
    p_value : Optional[float] = None
    n_a : int = 0
    n_b : int = 0
    n_subsampled : int = 0
    gamma : Optional[float] = None

@dataclass
class DistributionShift(Block):
    tests : dict = field(default_factory=dict)
    prompt_overlap_frac : Optional[float] = None

def mmd_test(
    A : np.ndarray, 
    B : np.ndarray, 
    name : str, 
    n_perm : int = 200,
    max_n : int = 2000, 
    min_n : int = 50, 
    seed : int = 0
) -> ShiftResult:
    sr = ShiftResult(name = name, n_a = len(A), n_b = len(B))
    
    if len(A) < min_n or len(B) < min_n:
        return _fail(sr, "unavailable", f"need {min_n} per side, got {len(A)} / {len(B)}")
    
    rng = np.random.default_rng(seed)
    take = lambda m : m[rng.choice(len(m), max_n, replace = False)] if len(m) > max_n else m
    A, B = take(A), take(B)
    m, k = len(A), len(B)
    sr.n_subsampled = m + k
    K, sr.gamma = _rbf_kernel(np.vstack([A, B]))
    
    z_0 = np.zeros((m + k, 1))
    z_0[: m] = 1.0
    
    sr.mmd2 = float(_mmd2_from_K(K, z_0, m, k)[0])

    Zind = np.zeros((m + k, n_perm))
    for b in range(n_perm):
        Zind[rng.permutation(m + k)[:m], b] = 1.0
    null = _mmd2_from_K(K, Zind, m, k)
    
    sr.p_value = float((1 + (null >= sr.mmd2).sum()) / (1 + n_perm))

    thin = sr.n_subsampled < (sr.n_a + sr.n_b)
    return _fail(sr, "approximate" if thin else "ok", f"subsampled to {sr.n_subsampled}" if thin else "")

def distribution_shift(
    train_pairs : Sequence[PairView], 
    val_pairs : Sequence[PairView], 
    emb_dict_train : dict, 
    emb_dict_val : dict,
    overlap_frac : Optional[float] = None,
    overlap_tau : float = 0.5, 
    **kw
) -> DistributionShift:
    ds = DistributionShift()
    if not val_pairs:
        return _fail(ds, "unavailable", "val split is not provided.")

    if overlap_frac is None:
        ptr = {p.prompt for p in train_pairs}
        pva = {p.prompt for p in val_pairs}
        overlap_frac = len(ptr & pva) / len(pva) if pva else 0.0
    
    ds.prompt_overlap_frac = overlap_frac

    for key in ("prompt", "chosen", "rejected"):
        ds.tests[key] = mmd_test(emb_dict_train[key], emb_dict_val[key], key, **kw)

    if ds.prompt_overlap_frac > overlap_tau:
        for t in ds.tests.values():
            _fail(t, "not_applicable", f"splits share {ds.prompt_overlap_frac:.1%} of val prompts, MMD is vacuous here")
        return _fail(ds, "not_applicable", f"prompt overlap {ds.prompt_overlap_frac:.1%}")
    
    return _fail(ds, "ok", "")

@dataclass
class NearDuplicates(Block):
    threshold : float = 0.95
    n_train : Optional[int] = None
    n_val : Optional[int] = None
    n_cross : Optional[int] = None
    rate_train : Optional[float] = None
    rate_cross : Optional[float] = None
    max_cross_cosine : Optional[float] = None
    nn_cosine_q50 : Optional[float] = None
    nn_cosine_q95 : Optional[float] = None
    n_compared : int = 0
    n_capped_from : int = 0
    nn_train : Optional[np.ndarray] = None

def _dup_scan(
    A : np.ndarray, 
    B : np.ndarray, 
    tau : float, 
    chunk : int = 1024,
    same : bool = False
):
    An, Bn = unit_normalize(A), unit_normalize(B)
    count, mx = 0, -1.0
    nn = np.full(len(A), -1.0)
    
    for i in range(0, len(A), chunk):
        S = An[i: i + chunk] @ Bn.T
        if same:
            rows = np.arange(S.shape[0])
            S[rows, i + rows] = -np.inf
        count += int((S > tau).sum())
        if S.size:
            mx = max(mx, float(S.max()))
            nn[i: i + chunk] = S.max(axis=1)
    if same:
        count //= 2
    return count, mx, nn

def unique_rows(texts : Sequence[str], X : np.ndarray) -> np.ndarray:
    seen, idx = set(), []
    for i, t in enumerate(texts):
        if t not in seen:
            seen.add(t)
            idx.append(i)
    return X[idx]

def near_duplicates(
    emb_train : np.ndarray, 
    emb_val : Optional[np.ndarray], 
    tau : float = 0.95,
    max_n : int = 10_000, 
    seed : int = 0
) -> NearDuplicates:

    nd = NearDuplicates(threshold=tau)
    if len(emb_train) < 2:
        return _fail(nd, "unavailable", "train has < 2-rows")

    rng = np.random.default_rng(seed)
    def cap(M):
        if M is None or len(M) <= max_n:
            return M, 0
        return M[rng.choice(len(M), max_n, replace=False)], len(M)

    A, from_a = cap(emb_train)
    B, _ = cap(emb_val)
    nd.n_capped_from = from_a
    nd.n_compared = len(A)

    nd.n_train, _, nn = _dup_scan(A, A, tau, same = True)
    nd.rate_train = nd.n_train / (len(A) * (len(A) - 1) / 2)
    nd.nn_cosine_q50, nd.nn_cosine_q95 = (float(v) for v in np.quantile(nn, [0.5, 0.95]))
    nd.nn_train = nn

    if B is not None and len(B) >= 2:
        nd.n_val, _, _ = _dup_scan(B, B, tau, same=True)
        nd.n_cross, nd.max_cross_cosine, _ = _dup_scan(A, B, tau)
        nd.rate_cross = nd.n_cross / (len(A) * len(B))

    return _fail(nd, "approximate" if from_a else "ok", f"capped from {from_a} to {max_n}" if from_a else "")

@dataclass
class Diversity(Block):
    vendi_prompt : Optional[float] = None
    vendi_chosen : Optional[float] = None
    vendi_rejected : Optional[float] = None
    n_rows : int = 0
    effective_sample_ratio : Optional[float] = None
    frac_isolated : Optional[float] = None

def diversity(
    emb : dict, 
    nn_train : Optional[np.ndarray] = None,
    isolated_tau : float = 0.5, 
    min_rows : int = 20
) -> Diversity:
    dv = Diversity(n_rows = len(emb.get("prompt", [])))
    if dv.n_rows < min_rows:
        return _fail(dv, "unavailable", f"only {dv.n_rows} rows")

    dv.vendi_prompt = vendi_score(emb["prompt"])
    dv.vendi_chosen = vendi_score(emb["chosen"])
    dv.vendi_rejected = vendi_score(emb["rejected"])
    dv.effective_sample_ratio = dv.vendi_prompt / dv.n_rows
    if nn_train is not None and len(nn_train):
        dv.frac_isolated = float((nn_train < isolated_tau).mean())
    return _fail(dv, "ok", "")

@dataclass
class ProfileReport:
    source : str = ""
    n_pairs : int = 0
    n_sampled_from : int = 0
    health : EmbeddingHealth = field(default_factory=EmbeddingHealth)
    separability : Separability = field(default_factory=Separability)
    delta : DeltaDirection = field(default_factory=DeltaDirection)
    grounding : Grounding = field(default_factory=Grounding)
    shift : DistributionShift = field(default_factory=DistributionShift)
    near_dup : NearDuplicates = field(default_factory=NearDuplicates)
    diversity : Diversity = field(default_factory=Diversity)
    warnings : list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_pairs": self.n_pairs,
            "n_sampled_from": self.n_sampled_from,
            "health": self.health.to_dict(),
            "separability": {
                **{k: v for k, v in self.separability.to_dict().items() if k != "slices"},
                "slices": {n: r.to_dict() for n, r in self.separability.slices.items()},
            },
            "delta": self.delta.to_dict(),
            "grounding": self.grounding.to_dict(),
            "shift": {
                **{k: v for k, v in self.shift.to_dict().items() if k != "tests"},
                "tests": {n: t.to_dict() for n, t in self.shift.tests.items()},
            },
            "near_dup": self.near_dup.to_dict(),
            "diversity": self.diversity.to_dict(),
            "warnings": self.warnings,
        }

def _embed_views(pairs, cfg : EmbedConfig, cache = None):
    n = len(pairs)
    texts = ([p.prompt for p in pairs] + [p.chosen for p in pairs] + [p.rejected for p in pairs])
    X, cache = embed(texts, cfg, cache)
    return {"prompt": X[: n], "chosen": X[n : 2 * n], "rejected": X[2 * n :]}, cache

def build_warnings(
    rep : ProfileReport, 
    ceiling : float = 0.75,
    one_axis_tau : float = 0.35, 
    controlled_tau : float = 0.05
) -> list[str]:

    w = []
    sep, dd, g, nd, dv = (rep.separability, rep.delta, rep.grounding, rep.near_dup, rep.diversity)

    if sep.frac_surface_controlled is not None and sep.frac_surface_controlled < controlled_tau:
        w.append(f"SURFACE DIFFERS BY CONSTRUCTION  frac_surface_controlled="
                 f"{sep.frac_surface_controlled:.3f} < {controlled_tau}. Chosen and rejected almost never "
                 f"share a format, so no slice can separate content from format.")

    if sep.usable and sep.separability_min is not None and sep.separability_min > ceiling:
        w.append(f"EVEN THE SHORTCUT FREE SLICE IS EASY  separability_min={sep.separability_min:.3f} "
                 f"> {ceiling:.2f}. The task stays trivial after every confound is controlled.")

    if sep.usable and sep.top_confound_drop and sep.top_confound_drop > 0.15:
        w.append(f"LARGE CONFOUND  controlling {sep.top_confound} dropped accuracy by "
                 f"{sep.top_confound_drop:+.3f}. That much of the signal was that one feature.")

    if dd.usable and dd.mean_pairwise_cosine and dd.mean_pairwise_cosine > one_axis_tau:
        w.append(f"ONE AXIS ONLY  mean_pairwise_cosine={dd.mean_pairwise_cosine:.3f}. "
                 f"The dataset teaches a single lesson. Read the pc1 examples.")

    if g.usable and g.frac_rejected_offtopic and g.frac_rejected_offtopic > 0.20:
        w.append(f"REJECTED OFF TOPIC  {g.frac_rejected_offtopic:.1%} of rejected responses "
                 f"are unrelated to their prompt. The model will learn to be a relevance detector.")

    if nd.usable and nd.n_cross:
        w.append(f"SEMANTIC LEAKAGE  {nd.n_cross:,} cross split near duplicate pairs at "
                 f"cosine > {nd.threshold}. Exact match checks miss these.")

    if dv.usable and dv.effective_sample_ratio and dv.effective_sample_ratio < 0.05:
        w.append(f"LOW DIVERSITY  effective_sample_ratio={dv.effective_sample_ratio:.3f}. "
                 f"{dv.n_rows:,} rows are really ~{dv.vendi_prompt:.0f} distinct examples.")
                 
    if rep.shift.status == "not_applicable":
        w.append(f"SHIFT TEST NOT MEANINGFUL  {rep.shift.reason}. "
                 f"Re-split at the prompt level and this test becomes informative.")
    return w

def profile(
    groups, 
    val_groups = None, 
    cfg : Optional[EmbedConfig] = None,
    source : str = "", 
    eps : float = 0.10, 
    tau : float = 0.95,
    seed : int = 0
) -> ProfileReport:

    cfg = cfg or EmbedConfig()
    rep = ProfileReport(source=source)
    pairs = groups_to_pairs(groups)
    rep.n_pairs = len(pairs)

    full_overlap = None
    
    if val_groups:
        prompt_train = {g.prompt for g in groups}
        prompt_val = {g.prompt for g in val_groups}
        full_overlap = len(prompt_train & prompt_val) / len(prompt_val) if prompt_val else 0.0

    if len(pairs) > cfg.max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), cfg.max_samples, replace=False)
        rep.n_sampled_from = len(pairs)
        pairs = [pairs[i] for i in idx]
        rep.n_pairs = len(pairs)

    if not pairs:
        rep.health = _fail(EmbeddingHealth(), "unavailable", "koi pair nahi")
        return rep

    E, cache = _embed_views(pairs, cfg)
    X = np.vstack([E["prompt"], E["chosen"], E["rejected"]])
    rep.health = embedding_health(X, cfg, cache, n_texts=len(X))
    if rep.health.status == "degenerate":
        rep.warnings = [f"EMBEDDINGS DEGENERATE  {rep.health.reason}. Baaki har metric is par khada hai isliye kuch nahi chalaya."]
        return rep

    masks = slice_masks(pairs, build_slice_specs(), eps=eps)
    rep.separability = separability(pairs, E["chosen"], E["rejected"], masks, seed=seed)
    rep.delta = delta_direction(pairs, E["chosen"], E["rejected"])
    rep.grounding = grounding(E["prompt"], E["chosen"], E["rejected"])

    Ptr = [p.prompt for p in pairs]
    Utr = unique_rows(Ptr, E["prompt"])

    if val_groups:
        vp = groups_to_pairs(val_groups)
        if len(vp) > cfg.max_samples:
            rng = np.random.default_rng(seed)
            vp = [vp[i] for i in rng.choice(len(vp), cfg.max_samples, replace=False)]
        Ev, _ = _embed_views(vp, cfg, cache)
        Uva = unique_rows([p.prompt for p in vp], Ev["prompt"])
        rep.shift = distribution_shift(pairs, vp, E, Ev, overlap_frac=full_overlap, seed=seed)
        rep.near_dup = near_duplicates(Utr, Uva, tau=tau, seed=seed)
    else:
        rep.shift = _fail(DistributionShift(), "unavailable", "val split diya hi nahi gaya")
        rep.near_dup = near_duplicates(Utr, None, tau=tau, seed=seed)

    rep.diversity = diversity(E, nn_train=rep.near_dup.nn_train)
    rep.warnings = build_warnings(rep)
    return rep

def _fmt(v, spec : str = ".3f", dash : str = "  n/a") -> str:
    return dash if v is None else format(v, spec)

def render(rep : ProfileReport, width : int = 76, n_slices : int = 8) -> str:
    L, bar = [], "=" * width
    sub = lambda t: f"\n{'-'*4} {t} {'-'*(width-len(t)-6)}"
    h, sep, dd, g, sh, nd, dv = (rep.health, rep.separability, rep.delta, rep.grounding, rep.shift, rep.near_dup, rep.diversity)

    L += [bar, "DATASET PROFILE   (tier 1: embeddings)", bar]
    
    L += [f"  source          : {rep.source or 'n/a'}",
          f"  pairs           : {rep.n_pairs:,}"
          + (f"   (sampled from {rep.n_sampled_from:,})" if rep.n_sampled_from else "")]

    L += [sub(f"EMBEDDINGS  [{h.status}]")]
    L += [f"  {h.backend} : {h.model_id}   dim {h.dim}",
          f"  texts {h.n_texts:,}   unique cached {h.n_unique:,}   "
          f"  hits {h.n_cache_hits:,}/{h.n_lookups:,}   embedded {h.n_embedded:,}",
          f"  mean_norm {_fmt(h.mean_norm)}   effective_rank {_fmt(h.effective_rank, '.1f')}"]
    if h.reason:
        L.append(f"  reason: {h.reason}")

    L += [sub(f"SEPARABILITY  [{sep.status}]")]
    if sep.usable:
        L.append(f"  {'slice':<26s}{'pairs':>9s}{'acc':>8s}{'auc':>8s}{'drop':>9s}")
        rows = sorted(sep.slices.values(),
                      key=lambda r: (-(r.drop_vs_raw or -9) if r.name != "raw" else -99))
        for r in rows[:n_slices]:
            if not r.usable:
                L.append(f"  {r.name:<26s}{r.n_pairs:>9,}{'':>8s}{'':>8s}   [{r.status}]")
                continue
            L.append(f"  {r.name:<26s}{r.n_pairs:>9,}{r.acc:>8.3f}{r.auc:>8.3f}"
                     f"{_fmt(r.drop_vs_raw, '+.3f', '     -'):>9s}")
        L += ["",
              f"  >> frac_surface_controlled = {_fmt(sep.frac_surface_controlled)}"
              f"   ({sep.slices['all_matched'].n_pairs:,} pairs where no surface feature differs)",
              f"  >> separability_raw = {_fmt(sep.separability_raw)}",
              f"  >> separability_min = {_fmt(sep.separability_min)}   <-- IMAANDAAR NUMBER",
              f"     top confound    = {sep.top_confound} "
              f"({_fmt(sep.top_confound_drop, '+.3f')})"]
    else:
        L.append(f"  reason: {sep.reason}")

    L += [sub(f"DELTA DIRECTION  [{dd.status}]")]
    if dd.usable:
        L += [f"  mean_pairwise_cosine  {_fmt(dd.mean_pairwise_cosine)}",
              f"  frac_aligned_w_mean   {_fmt(dd.frac_aligned_with_mean)}",
              f"  pc explained var      {[round(v, 3) for v in dd.pc_explained_var]}",
              f"  effective_dim         {_fmt(dd.effective_dim, '.1f')}   (unit arrows {_fmt(dd.effective_dim_unit, '.1f')})",
              f"  delta norm            q05 {_fmt(dd.delta_norm_q05, '.2f')}"
              f"   q50 {_fmt(dd.delta_norm_q50, '.2f')}   q95 {_fmt(dd.delta_norm_q95, '.2f')}"]
        if dd.pc1_top:
            L += ["", "  PC1 ke dono sire  (yahi axis model seekhega)"]
            for lbl, ex in (("  +", dd.pc1_top[:2]), ("  -", dd.pc1_bottom[:2])):
                for c, r in ex:
                    L.append(f"  {lbl} chosen   {c[:60]!r}")
                    L.append(f"      rejected {r[:60]!r}")
    else:
        L.append(f"  reason: {dd.reason}")

    L += [sub(f"GROUNDING  [{g.status}]")]
    if g.usable:
        L += [f"  cos(prompt, chosen)   {_fmt(g.cos_prompt_chosen)}",
              f"  cos(prompt, rejected) {_fmt(g.cos_prompt_rejected)}",
              f"  relevance_gap         {_fmt(g.relevance_gap, '+.3f')}",
              f"  rejected off topic    {_fmt(g.frac_rejected_offtopic, '.1%')}"]
    else:
        L.append(f"  reason: {g.reason}")

    L += [sub(f"DISTRIBUTION SHIFT  [{sh.status}]")]
    L.append(f"  prompt overlap  {_fmt(sh.prompt_overlap_frac, '.1%')}")
    for name, t in sh.tests.items():
        L.append(f"  {name:<12s}mmd2 {_fmt(t.mmd2, '.5f')}   p {_fmt(t.p_value)}"
                 f"   [{t.status}]")
    if sh.reason:
        L.append(f"  reason: {sh.reason}")

    L += [sub(f"NEAR DUPLICATES  [{nd.status}]  tau={nd.threshold}")]
    L += [f"  within train    {_fmt(nd.n_train, ',')}   rate {_fmt(nd.rate_train, '.4f')}",
          f"  within val      {_fmt(nd.n_val, ',')}",
          f"  cross split     {_fmt(nd.n_cross, ',')}   max cos {_fmt(nd.max_cross_cosine)}",
          f"  nn cosine       q50 {_fmt(nd.nn_cosine_q50)}   q95 {_fmt(nd.nn_cosine_q95)}",
          f"  compared        {nd.n_compared:,}"
          + (f"   (capped from {nd.n_capped_from:,})" if nd.n_capped_from else "")]

    L += [sub(f"DIVERSITY  [{dv.status}]")]
    L += [f"  vendi prompt    {_fmt(dv.vendi_prompt, '.1f')}   of {dv.n_rows:,} rows",
          f"  vendi chosen    {_fmt(dv.vendi_chosen, '.1f')}",
          f"  vendi rejected  {_fmt(dv.vendi_rejected, '.1f')}",
          f"  eff sample rate {_fmt(dv.effective_sample_ratio, '.4f')}",
          f"  frac isolated   {_fmt(dv.frac_isolated, '.3f')}"]

    L += [sub("WARNINGS")]
    L += [f"  !  {w}" for w in rep.warnings] or ["  none"]
    L += ["", bar]
    return "\n".join(L)
