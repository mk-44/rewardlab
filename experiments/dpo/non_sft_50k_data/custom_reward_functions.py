from __future__ import annotations
import re
import string
from typing import List, Sequence

import pronouncing
from rlhf.core.scoring.rewards import reward_func
from rlhf.reward.scoring import from_run

RM_RUN_DIR = "experiments/reward/rhyme_50k_distilbert/20260905_025233_e4"
RM_WHICH = "best"
RM_DEVICE = "cpu"
RM_BATCH_SIZE = 64

_PUNCT = str.maketrans("", "", string.punctuation)


def _last_word(line : str) -> str:
    words = line.translate(_PUNCT).lower().split()
    return words[-1] if words else ""


def _rhyme_key(word : str) -> str:
    phones = pronouncing.phones_for_word(word)
    if not phones:
        return ""
    return re.sub(r"\d", "", pronouncing.rhyming_part(phones[0]))


def _couplet_score(response : str) -> float:
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0
    a, b = _last_word(lines[0]), _last_word(lines[1])
    if not a or not b or a == b:
        return 0.0
    ka, kb = _rhyme_key(a), _rhyme_key(b)
    return float(bool(ka) and ka == kb)


@reward_func(name = "rhyme_rate")
def rhyme_rate(prompts : Sequence[str], responses : Sequence[str]) -> List[float]:
    return [_couplet_score(r) for r in responses]


reward_model = from_run(RM_RUN_DIR, device = RM_DEVICE, batch_size = RM_BATCH_SIZE, which = RM_WHICH, name = "reward_model")
