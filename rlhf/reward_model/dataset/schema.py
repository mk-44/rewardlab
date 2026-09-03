from __future__ import annotations
import math
import random
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Any, Literal
from rlhf.reward_model.core.contracts import ConfigError, DataError

PairPolicy = Literal["all_pairs", "adjacent", "best_vs_rest"]
UNASSIGNED_UID = -1

@dataclass(frozen = True)
class PairView:
    "One chosen versus rejected pair taken out of a group."
    prompt : str
    chosen : str
    rejected : str

    group_uid : int
    chosen_idx : int
    rejected_idx : int

    slices : dict[str, str] = field(default_factory = dict)
    margin_hint : Optional[float] = None

    @property
    def uid(self) -> tuple[int, int, int]:
        "Globally unique identity. Needed to dedup after a distributed gather."
        return (self.group_uid, self.chosen_idx, self.rejected_idx)
    

@dataclass(frozen = True)
class PreferenceGroup:
    "K responses for one prompt with a ranking. Lower rank means better."
    prompt : str
    responses : list[str]

    ranking : Optional[list[int]] = None
    scores : Optional[list[float]] = None

    slices : dict[str, str] = field(default_factory = dict)
    meta : dict[str, Any] = field(default_factory = dict)

    uid : int = UNASSIGNED_UID

    def __post_init__(self) -> None:
        "Structural checks only. Data quality is the audit step job."
        if not isinstance(self.prompt, str):
            raise DataError(f"prompt must be str, got {type(self.prompt).__name__}.")

        if not isinstance(self.responses, (list, tuple)):
            raise DataError(f"responses must be a list, got {type(self.responses).__name__}.")
        
        if len(self.responses) < 2:
            raise DataError(f"A group needs at least 2 responses, got {len(self.responses)}. prompt={self.prompt[:60]!r}")
        
        for i, r in enumerate(self.responses):
            if not isinstance(r, str):
                raise DataError(    f"responses[{i}] must be str, got {type(r).__name__}.")

        K = len(self.responses)

        if self.ranking is not None:
            if len(self.ranking) != K:
                raise DataError(f"ranking has {len(self.ranking)} entries but there are {K} responses — they must match one-to-one.") 
            for i, r in enumerate(self.ranking):
                if not isinstance(r, int) or isinstance(r, bool):
                    raise DataError(f"ranking[{i}] must be int, got {type(r).__name__}.")
                if r < 0:
                    raise DataError(f"ranking[{i}] must be >= 0, got {r}.")
        
        if self.scores is not None:
            if len(self.scores) != K:
                raise DataError(f"scores has {len(self.scores)} entries but there are {K} responses — they must match one-to-one.")
            for i, s in enumerate(self.scores):
                if not isinstance(s, (int, float)) or isinstance(s, bool):
                    raise DataError(f"scores[{i}] must be a number, got {type(s).__name__}.")
                if not math.isfinite(float(s)):
                    raise DataError(f"scores[{i}] must be finite, got {s}.")
        
        if not isinstance(self.slices, dict):
            raise DataError(f"slices must be a dict, got {type(self.slices).__name__}.")
        
        for k, v in self.slices.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise DataError(f"slices must be dict[str, str]. Bad entry: {k!r}={v!r} ({type(k).__name__} -> {type(v).__name__}). Wrap the value in str() at load time.")
        
    @property
    def K(self) -> int:
        "How many responses."
        return len(self.responses)

    @property
    def has_ranking(self) -> bool:
        "True when an explicit ranking was given."
        return self.ranking is not None

    @property
    def has_scores(self) -> bool:
        "True when graded scores were given."
        return self.scores is not None

    @property
    def has_uid(self) -> bool:
        "True once the loader has handed out a global id."
        return self.uid != UNASSIGNED_UID

    @property
    def effective_ranking(self) -> Optional[list[int]]:
        "Ranking from the field or derived from scores. None when neither exists."
        if self.ranking is not None:
            return list(self.ranking)
        if self.scores is not None:
            return ranks_from_scores(self.scores)
        return None

    @property
    def best_index(self) -> Optional[int]:
        "Index of the best response. Used by Best of N eval."
        ranks = self.effective_ranking
        if ranks is None:
            return None
        return min(range(self.K), key = lambda i: ranks[i])

    @property
    def has_preference_signal(self) -> bool:
        "False when all responses are tied so no pair can be built."
        ranks = self.effective_ranking
        if ranks is None:
            return False
        first = ranks[0]
        return any(r != first for r in ranks)
    
    def _rank_buckets(self) -> tuple[list[int], dict[int, list[int]]]:
        "Bucket indices by rank level in one pass. Every policy starts here. [O(K + L log L)]"
        ranks = self.effective_ranking
        if ranks is None:
            raise DataError(f"Cannot build pairs: group uid={self.uid} has neither `ranking` nor `scores`. One of them is required.")
        buckets: dict[int, list[int]] = {}
        for i, r in enumerate(ranks):
            buckets.setdefault(r, []).append(i)
        return sorted(buckets), buckets
    
    def _make_pair(self, i: int, j: int) -> PairView:
        "Build one PairView. Slices are shared and not copied."
        return PairView(
            prompt = self.prompt,
            chosen = self.responses[i],
            rejected = self.responses[j],
            group_uid = self.uid,
            chosen_idx = i,
            rejected_idx = j,
            slices = self.slices,
            margin_hint = (float(self.scores[i] - self.scores[j]) if self.scores is not None else None)
        )
    
    def iter_pairs(self, policy : PairPolicy = "all_pairs"):
        "Stream pairs one at a time. [O(K + L log L + P) time and O(1) memory]"
        levels, buckets = self._rank_buckets()
        L = len(levels)

        if policy == "all_pairs":
            for a_idx in range(L):
                for b_idx in range(a_idx + 1, L):
                    for i in buckets[levels[a_idx]]:
                        for j in buckets[levels[b_idx]]:
                            yield self._make_pair(i, j)

        elif policy == "best_vs_rest":
            for i in buckets[levels[0]]:
                for b_idx in range(1, L):
                    for j in buckets[levels[b_idx]]:
                        yield self._make_pair(i, j)

        elif policy == "adjacent":
            for a_idx in range(L - 1):
                for i in buckets[levels[a_idx]]:
                    for j in buckets[levels[a_idx + 1]]:
                        yield self._make_pair(i, j)
        else:
            raise ConfigError(f"Unknown pair policy {policy!r}. Valid: all_pairs, adjacent, best_vs_rest.")

    
    def to_pairs(self, policy : PairPolicy = "all_pairs", max_pairs : Optional[int] = None, rng : Optional[random.Random] = None) -> list[PairView]:
        "Same as iter_pairs but returns a list. max_pairs needs an rng."
        if max_pairs is None:
            return list(self.iter_pairs(policy))
        
        if max_pairs < 0:
            raise ConfigError(f"max_pairs must be >= 0, got {max_pairs}.")

        if rng is None:
            raise ConfigError(f"max_pairs={max_pairs} needs an `rng`. Taking the first N would always pick the top-ranked pairs — a biased sample. Pass random.Random(seed).")

        return reservoir_sample(self.iter_pairs(policy), max_pairs, rng)
    
    def n_pairs(self, policy : PairPolicy = "all_pairs"):
        "Count pairs by closed form without building any. [O(K + L log L)]"
        levels, buckets = self._rank_buckets()
        K = self.K
        sizes = [len(buckets[l]) for l in levels]

        if policy == "all_pairs":
            return ((K * K) - sum(s * s for s in sizes)) // 2
        elif policy == "best_vs_rest":
            return sizes[0] * (K - sizes[0])
        elif policy == "adjacent":
            res = 0
            for i in range(len(sizes) - 1):
                res += (sizes[i] * sizes[i + 1])
            return res
        raise ConfigError(f"Unknown pair policy {policy!r}. Valid: all_pairs, adjacent, best_vs_rest.")
    
    @classmethod
    def from_pairwise(
        cls,
        prompt : str,
        chosen : str,
        rejected : str,
        slices : Optional[dict[str, str]] = None,
        meta : Optional[dict[str, Any]] = None,
        uid : int = UNASSIGNED_UID
    ) -> PreferenceGroup:
        "Lift a chosen versus rejected row into canonical K way form."
        return cls(
            prompt = prompt,
            responses = [chosen, rejected],
            ranking = [0, 1],
            slices = slices or {},
            meta = meta or {},
            uid = uid
        )

    def with_uid(self, uid: int) -> PreferenceGroup:
        "Return a copy holding a new uid because the class is frozen."
        return replace(self, uid = uid)


def reservoir_sample(stream, max_num : int, rng : random.Random):
    "Pick max_num items uniformly from a stream. [O(P) time and O(max_num) memory]"
    if max_num <= 0:
        return []
    out = []
    for i, item in enumerate(stream):
        if i < max_num:
            out.append(item)
        else:
            j = rng.randrange(i + 1)
            if j < max_num:
                out[j] = item

    out.sort(key=lambda p: (p.chosen_idx, p.rejected_idx))
    return out



def ranks_from_scores(scores : Sequence[float]) -> list[int]:
    "Higher score means better. Ties share a rank in competition style."
    order = sorted(range(len(scores)), key = lambda i : -scores[i])
    ranks = [0] * len(scores)
    curr = 0

    for pos, i in enumerate(order):
        if pos > 0 and scores[i] < scores[order[pos - 1]]:
            curr = pos
        ranks[i] = curr
    return ranks

def assign_uids(groups : Sequence[PreferenceGroup], start : int = 0) -> list[PreferenceGroup]:
    "Give every group a global id before any sharding happens."
    return [g.with_uid(start + i) for i, g in enumerate(groups)]


def iter_group_pairs(
    groups : Sequence[PreferenceGroup],
    policy : PairPolicy = "all_pairs",
    max_pairs_per_group : Optional[int] = None,
    seed : Optional[int] = None
):
    "Stream pairs over the whole dataset. [O(1) memory per group]"
    rng = random.Random(seed) if seed is not None else None
    for g in groups:
        if max_pairs_per_group is None:
            yield from g.iter_pairs(policy = policy)
        else:
            yield from g.to_pairs(policy = policy, max_pairs = max_pairs_per_group, rng = rng)

def groups_to_pairs(
    groups : Sequence[PreferenceGroup],
    policy : PairPolicy = "all_pairs",
    max_pairs_per_group : Optional[int] = None,
    seed : Optional[int] = None
) -> list[PairView]:
    "Same but materialised. Watch memory on large datasets."
    return list(iter_group_pairs(groups, policy, max_pairs_per_group, seed))


def count_group_pairs(
    groups : Sequence[PreferenceGroup],
    policy : PairPolicy = "all_pairs",
) -> int:
    "Count dataset pairs without building any. [O(sum of K)]"
    return sum(g.n_pairs(policy) for g in groups)


