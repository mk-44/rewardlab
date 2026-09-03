# Negative result: the rhyming preference set is a generator fingerprint, not a preference

**Status: retired for reward-model training. Retained as a diagnostic case study.**

This folder holds the cross-cutting analysis of the rhyming preference data. The two
training runs it refers to live in sibling folders:

| Run folder | What it is |
|---|---|
| [`../distill_bert/`](../distill_bert/) | `distilbert-base-uncased`, full finetune, prompt + response |
| [`../probe_minilm/`](../probe_minilm/) | `all-MiniLM-L6-v2`, backbone frozen, linear head, response only |

The finding in one sentence: **chosen and rejected responses were produced by two
different processes, so the preference label is recoverable from surface format alone,
and no reward model trained on it can learn anything about rhyme quality.**

Every number below came out of this repo's own CLI. Reproduction commands are in
[section 7](#7-reproducing-every-number).

---

## 1. Provenance

Raw source, kept outside the repository:

```
../RL/projects/rhyming/data/preferences_train.json   sha256 b155cf14…79a6809
../RL/projects/rhyming/data/preferences_val.json     sha256 99a71c55…4c69eee
```

The distributed split was discarded and rebuilt at the **prompt** level, because the
original split shared prompts across train and val. After the rebuild:

| Quantity | Value |
|---|---|
| Rows in | 49,948 |
| Unique prompts in | 1,058 |
| Exact duplicate rows found | 80 |
| Train rows / prompts | 43,659 / 925 |
| Val rows / prompts | 6,289 / 133 |
| Prompt overlap between splits | 0 |
| Rows per prompt | 32 to 50 |
| Group size K | 2 for every group |

Recorded in [`../../../preference_data/reports/split_report.json`](../../../preference_data/reports/split_report.json).

Note the shape: **1,058 prompts carrying 49,948 pairs**, roughly 47 pairs per prompt.
Effective sample size is far below the row count, and every result below should be read
with 925 independent prompts in mind rather than 43,659 independent examples.

---

## 2. Tier 0: the label is readable without a model

No embeddings, no training. Pairwise accuracy of single-feature and bag-of-words
classifiers on the train split:

| Feature the classifier is allowed to see | Pairwise accuracy |
|---|---|
| Bag of words, naive Bayes | **0.998** |
| Character length | 0.946 |
| Newline count | 0.926 |
| Punctuation count | 0.924 |
| Word count | 0.880 |
| Average word length | 0.523 |
| Unique word ratio | 0.402 |

The declared expected ceiling for the task was 0.75, the band where human-annotated
preference data normally sits. A naive Bayes classifier over raw tokens beat that
ceiling by 0.248. That single number is enough to stop the project, and the audit
raised it as a regime mismatch before any GPU time was spent.

**Surface asymmetry** is where the mechanism becomes visible:

| Feature | Present in chosen | Present in rejected | Gap |
|---|---|---|---|
| `has_newline` | 1.000 | 0.148 | **+0.852** |
| `multi_line` | 1.000 | 0.148 | **+0.852** |
| `has_comma` | 0.993 | 0.294 | +0.699 |
| `ends_with_period` | 0.930 | 0.982 | −0.052 |
| `exclamation` | 0.058 | 0.007 | +0.051 |
| `question_mark` | 0.026 | 0.009 | +0.017 |
| `hedging` | 0.000 | 0.010 | −0.010 |

Every chosen response in the dataset contains a newline. Roughly one rejected response
in seven does. Chosen responses are formatted verse. Rejected responses are
single-line prose. A classifier that outputs "chosen if the text contains `\n`"
scores about 0.93 on its own.

**Length** points the same way:

| | q05 | median | q95 |
|---|---|---|---|
| Chosen, characters | 76 | 91 | 107 |
| Rejected, characters | 17 | 54 | 91 |

P(chosen is longer) is 0.943. Only 10.7 percent of pairs are near-equal in length,
which is the entire length-controlled evaluation slice.

**The discriminative vocabulary** confirms the two-generator reading:

- Predicts chosen: `ignite`, `kindness`, `spirit`, `collide`, `swirl`, `i'd`, `nineteen`, `soul`, `you'll`, `courage`, `challenge`, `worries`
- Predicts rejected: `rat`, `likely`, `fat`, `located`, `practicing`, `mat`, `due`, `typically`, `involves`, `popular`, `sat`, `includes`

The chosen list is poetic register. The rejected list mixes encyclopedic prose
(`typically`, `includes`, `involves`, `located`) with nursery-rhyme cliché
(`rat`, `fat`, `mat`, `sat`). Note carefully that `rat`, `fat`, `mat` and `sat`
**do rhyme**. The dataset is not separating rhyming text from non-rhyming text. It is
separating one generator's output style from another's.

Six warnings fired on this dataset: regime mismatch, lexically trivial, trivial model
already wins, length imbalance, strong surface shortcut, and no slice keys.

---

## 3. Tier 1: controlling the shortcuts does not help

Embedding the responses with `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions,
on a 2,000-pair sample. A logistic separator on the embedding difference:

| Slice | Pairs | Accuracy | AUC | Drop from raw |
|---|---|---|---|---|
| raw | 2,000 | 0.989 | 0.996 | — |
| `match_unique_word_ratio` | 1,557 | 0.981 | 0.996 | +0.008 |
| `match_has_comma` | 606 | 0.983 | 0.999 | +0.006 |
| `match_exclamation` | 1,889 | 0.983 | 0.998 | +0.005 |
| `match_avg_word_length` | 772 | 0.987 | 0.998 | +0.002 |
| `match_ends_with_period` | 1,850 | 0.989 | 1.000 | −0.000 |
| `match_question_mark` | 1,948 | 0.992 | 0.996 | −0.003 |
| `match_hedging` | 1,980 | 0.992 | 0.999 | −0.004 |

Separability after controlling the worst confound is **0.981**. Controlling shortcuts
buys 0.008. The task stays trivial.

The number that explains why:

```
frac_surface_controlled = 0.004      7 pairs out of 2,000
```

In only seven pairs did chosen and rejected share every surface feature. **A
format-controlled evaluation set cannot be built from this data, because the format is
the label.** There is no subpopulation large enough to ask the question "does the model
prefer better rhymes when formatting is held constant."

**Grounding is inverted**, which is the most damaging single finding:

| Measurement | Value |
|---|---|
| cos(prompt, chosen) | 0.469 |
| cos(prompt, rejected) | **0.583** |
| Relevance gap | **−0.115** |
| Rejected responses off-topic | 17.4% |

The rejected response is on average *more* semantically related to the prompt than the
chosen one. A reward model fit to this data is trained to assign higher reward to text
that answers the question less directly. Used as a PPO or GRPO reward source, it
actively pushes the policy off-topic. This alone disqualifies the data.

Supporting structure:

| Measurement | Value | Reading |
|---|---|---|
| PC explained variance | [0.054, 0.045, 0.020] | No dominant axis. The difference is spread across many directions. |
| Effective dimension of deltas | 94.5 | Same reading. |
| Mean pairwise cosine of deltas | 0.041 | Individual differences are near-orthogonal. |
| Fraction aligned with mean delta | 0.970 | Yet almost all of them share one sign along the mean. |
| Embedding effective rank | 57.6 | |
| Vendi diversity, prompts | 125.2 of 2,000 rows | Effective sample rate 0.063. |
| Cross-split near-duplicates at cos > 0.95 | 3, max 0.984 | Exact-match checks miss these. |

The combination of low PC1 variance with 0.970 mean-alignment is worth pausing on. No
single direction dominates, but nearly every pair points to the same side of the mean
difference. That is the signature of a broad, consistent distribution shift between the
two response sets rather than a single feature axis.

---

## 4. Tier 2, experiment A: DistilBERT full finetune

Configuration in [`../distill_bert/config.yaml`](../distill_bert/config.yaml).
Backbone `distilbert-base-uncased`, mean pooling, `max_length` 128, prompt and response
both visible, 2 epochs, batch 16, learning rate 2e-5, cosine schedule, 200 warmup steps,
gradient clip 1.0, MPS, float32, seed 0.

| Result | Value |
|---|---|
| Validation accuracy | **0.99968**, standard error 0.00022 |
| Pairs evaluated | 6,289, all out-of-sample prompts |
| Best checkpoint | step 1,000 of a 5,458-step budget |
| Early stopped | step 3,000 |
| Wall clock | 942 s |
| Loss | 0.625 to 0.00035 |
| Mean margin | 12.63, median 13.13, p10 9.73, p90 14.84 |
| Mean chosen reward | +6.23 |
| Mean rejected reward | −6.39 |

Two pairs wrong out of 6,289.

**Every slice in the Tier-1 registry**, sorted worst first:

| Slice | Pairs | Accuracy | SE |
|---|---|---|---|
| `match_newline_count` | 937 | 0.998 | 0.002 |
| `match_has_newline` | 938 | 0.998 | 0.002 |
| `match_multi_line` | 938 | 0.998 | 0.002 |
| `match_char_length` | 745 | 0.999 | 0.001 |
| `match_has_comma` | 1,826 | 0.999 | 0.001 |
| all 17 remaining slices | up to 6,289 | 1.000 | 0.000 |

The model did not even need the length shortcut. Holding character length fixed leaves
accuracy at 0.9987. Holding newline count fixed leaves it at 0.9979. It read the
distribution difference directly, and that difference survives every control the
registry can impose.

Direction geometry: cos(w, mean delta) = +0.924. The own-space first principal
component absorbs 89.4 percent of unit-delta variance, but that number describes the
finetuned model's own representation space, not the Tier-1 instrument space, so it is
**not** comparable to Tier-1's 0.054. The ledger flags this explicitly. Comparing them
would be the mistake the probe run was designed to avoid.

---

## 5. Tier 2, experiment B: the frozen probe

This run exists to make the Tier-1 comparison legitimate. The backbone is frozen and is
the *same* encoder the Tier-1 instrument used, with the same pooling and the same
`{response}` template, so the learned head weight and the Tier-1 principal component
live in one vector space and their cosine is meaningful.

Configuration in [`../probe_minilm/config.yaml`](../probe_minilm/config.yaml).
`all-MiniLM-L6-v2` frozen, mean pooling, `max_length` 512, template `{response}`,
3 epochs, batch 32, learning rate 1e-3, cosine schedule, 50 warmup steps, MPS, float32,
seed 0.

| Result | Value |
|---|---|
| Validation accuracy | **0.98728**, standard error 0.00141 |
| Best checkpoint | step 3,500 of 4,095, full budget run |
| Wall clock | 179 s |
| Loss | 0.644 to 0.0426 |
| Mean margin | 5.09, median 5.06, p10 2.06, p90 8.19 |
| cos(head weight, Tier-1 PC1) | **+0.735** |
| cos(head weight, mean delta) | +0.825 |
| Tier-1 PC1 share of unit-delta variance | 0.055 |

**A frozen off-the-shelf sentence encoder with a single linear layer on top reaches
0.987 in three minutes.** No representation learning occurred. The label direction was
already present in a generic embedding space trained by someone else on unrelated data.

The +0.735 cosine is the direct measurement the whole probe was built for. The axis
that Tier-1 found without ever seeing a label, purely as the principal component of
embedding differences, is the same axis the supervised head learned. The dataset's
preference signal and its dominant unsupervised artifact axis are the same thing.

Unlike the finetune, this model **does** lean on length:

| Slice | Pairs | Accuracy | SE | Drop from raw |
|---|---|---|---|---|
| `match_word_count` | 526 | 0.954 | 0.009 | −0.033 |
| `match_char_length` | 745 | 0.966 | 0.007 | −0.021 |
| `match_newline_count` | 937 | 0.971 | 0.005 | −0.016 |
| `match_punctuation` | 514 | 0.977 | 0.007 | −0.010 |
| raw | 6,289 | 0.987 | 0.001 | — |

Holding word count fixed costs the probe 3.3 points. The capacity-limited model reaches
for the cheapest correlate available. The full finetune had enough capacity to encode
the richer format signature and did not need to.

---

## 6. Why this ends the dataset, and what survives

### Three independent disqualifications

**No headroom.** A bag-of-words classifier scores 0.998. The finetune scores 0.99968.
The entire contribution of a transformer, a preference objective, and 942 seconds of
training is 0.0017 over naive Bayes. There is no measurable quantity left for a reward
model to be good or bad at.

**The signal does not transfer to the setting it is for.** A reward model exists to rank
samples from a policy. Every sample from a rhyming policy will be multi-line verse with
commas. The format tell, worth 0.93 on its own and effectively all of the remaining
signal, is constant across the candidates it would have to rank. On its actual job
distribution this model has no signal at all.

**The reward direction is anti-correlated with task success.** Chosen responses are less
related to their prompts than rejected ones, gap −0.115, with 17.4 percent of rejected
responses off-topic. Optimizing a policy against this reward pushes it away from
answering the prompt.

### What is not wrong with it

The pipeline. Every one of these findings was produced by the repo's own tooling,
before and after training, and the ledger predicted the Tier-2 outcomes from Tier-0 and
Tier-1 numbers alone. The audit fired six warnings on the first pass in under ten
seconds. The cost of this negative result was about 19 minutes of laptop compute.

The task. Rhyming has a property that almost no public preference dataset has:
**computable ground truth.** Whether two lines rhyme is decidable by phoneme lookup with
no model in the loop. That makes it an unusually good controlled testbed for reward
hacking, because the true objective and the learned proxy can be measured separately and
the gap between them plotted directly during policy optimization.

### How to regenerate it correctly

The defect is that chosen and rejected came from different generators. The fix is to
make them come from the same one:

1. Sample K responses per prompt from a **single** policy at a single temperature, so
   format, length distribution and register are held constant by construction.
2. Label with a **programmatic rhyme scorer**, phoneme match on line endings, with a
   documented tiebreak for equal scores.
3. Re-run the audit and require `frac_surface_controlled` above 0.05, `lexical_nb`
   below 0.75, and `relevance_gap` at or above zero before training anything.
4. Keep the ground-truth scorer out of the reward model's inputs so it stays an
   independent measurement of hacking.

Gate 3 is the point. The audit that condemned this dataset is the same audit that would
clear its replacement, and the thresholds are written down in advance.

---

## 7. Reproducing every number

All commands run from the repository root.

```bash
# Rebuild the prompt-level split from the raw source outside the repo
python -m rlhf resplit --config preference_data/data.yaml \
    --set data.train_path=../RL/projects/rhyming/data/preferences_train.json \
    --set data.val_path=../RL/projects/rhyming/data/preferences_val.json

# Tier 0, seconds, no model
python -m rlhf audit   --config preference_data/data.yaml

# Tier 1, needs the embedding backend
python -m rlhf profile --config preference_data/data.yaml

# Tier 2, the two training runs
python -m rlhf train --config experiments/reward_model/distill_bert/config.yaml \
    --set run_name=20260902_0033_e2_base
python -m rlhf train --config experiments/reward_model/probe_minilm/config.yaml \
    --set run_name=20260902_2343_e3_probe

# Evaluation and the prediction-versus-outcome ledger
python -m rlhf eval   --config experiments/reward_model/probe_minilm/config.yaml \
    --set run_name=20260902_2343_e3_probe
python -m rlhf ledger --config experiments/reward_model/probe_minilm/config.yaml \
    --set run_name=20260902_2343_e3_probe
```

Artifacts:

| File | Contents |
|---|---|
| [`../../../preference_data/reports/audit_report.txt`](../../../preference_data/reports/audit_report.txt) | Tier 0, section 2 |
| [`../../../preference_data/reports/profile_report.txt`](../../../preference_data/reports/profile_report.txt) | Tier 1, section 3 |
| [`../distill_bert/20260902_0033_e2_base/tier2_report.txt`](../distill_bert/20260902_0033_e2_base/tier2_report.txt) | Tier 2 A, section 4 |
| [`../probe_minilm/20260902_2343_e3_probe/tier2_report.txt`](../probe_minilm/20260902_2343_e3_probe/tier2_report.txt) | Tier 2 B, section 5 |
| [`../distill_bert/ledger.md`](../distill_bert/ledger.md), [`../probe_minilm/ledger.md`](../probe_minilm/ledger.md) | Predictions versus outcomes |

Checkpoints are not committed. Both runs reproduce from the configs above on Apple
Silicon in 942 s and 179 s respectively.
