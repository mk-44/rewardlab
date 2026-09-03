# rewardlab

A from-scratch RLHF stack, written to be inspected rather than imported. Reward models
first, then DPO, PPO and GRPO. Every component is implemented directly on PyTorch with
no training framework underneath it, and the numerical decisions that usually stay
buried are made explicit and tested.

The part worth your attention is not the trainer. It is the **diagnostic stack**: a
three-tier instrument that tells you whether your preference data can support a reward
model *before* you train one, and then checks the trained model against those
predictions. The rest of this README is the story of the first dataset it was pointed
at, because that dataset failed and the failure is more instructive than a success
would have been.

```bash
python -m rlhf audit --config preference_data/data.yaml    # 10 seconds, no model, no GPU
```

---

## The headline result: a dataset that scored 0.9997 and was worthless

A reward model trained on this project's original rhyming preference data reached
**99.968 percent** pairwise validation accuracy on held-out prompts, with zero prompt
leakage and honest evaluation. That number is not a success. It is the symptom.

The dataset's chosen and rejected responses had been produced by two different
generation processes. The preference label was therefore recoverable from surface
format alone, and the reward model learned to detect the generator, not to judge the
rhyme. This section is the full evidence chain. The complete forensic record, with
every table and reproduction command, is in
[`experiments/reward_model/rhyming/NEGATIVE_RESULT.md`](experiments/reward_model/rhyming/NEGATIVE_RESULT.md).

### The data

49,948 preference pairs over 1,058 prompts, re-split at the prompt level because the
original split shared prompts across train and validation.

| Quantity | Value |
|---|---|
| Train rows / prompts | 43,659 / 925 |
| Validation rows / prompts | 6,289 / 133 |
| Prompt overlap between splits | 0 |
| Rows per prompt | 32 to 50 |

Roughly 47 pairs per prompt, so the effective sample size is far below the row count.

### Tier 0 caught it in ten seconds, before any training

No embeddings, no model. Pairwise accuracy of classifiers restricted to a single
feature, or to raw token counts:

| What the classifier may see | Pairwise accuracy |
|---|---|
| Bag of words, naive Bayes | **0.998** |
| Character length | 0.946 |
| Newline count | 0.926 |
| Punctuation count | 0.924 |
| Word count | 0.880 |
| Average word length | 0.523 |
| Unique word ratio | 0.402 |

The declared expected ceiling for the task was 0.75, the band where human-annotated
preference data normally sits. Naive Bayes over raw tokens cleared it by 0.248. The
audit raised this as a regime mismatch immediately: human preferences are not this
separable, so either the declared regime was wrong or a shortcut was doing the work.

The mechanism is visible in the surface asymmetry table:

| Feature | Present in chosen | Present in rejected | Gap |
|---|---|---|---|
| `has_newline` | 1.000 | 0.148 | **+0.852** |
| `multi_line` | 1.000 | 0.148 | **+0.852** |
| `has_comma` | 0.993 | 0.294 | +0.699 |

Every chosen response contains a newline. About one rejected response in seven does.
Chosen responses are formatted verse; rejected responses are single-line prose. The
rule "chosen if the text contains a line break" scores roughly 0.93 by itself.

Length tells the same story. Chosen responses have a median of 91 characters against 54
for rejected, P(chosen is longer) is 0.943, and only 10.7 percent of pairs are close
enough in length to form a length-controlled evaluation slice.

The discriminative vocabulary settles the interpretation:

- **Predicts chosen:** `ignite`, `kindness`, `spirit`, `collide`, `swirl`, `soul`, `courage`, `challenge`, `worries`
- **Predicts rejected:** `rat`, `fat`, `mat`, `sat`, `typically`, `includes`, `involves`, `located`, `likely`, `due`

The chosen list is poetic register. The rejected list mixes encyclopedic prose with
nursery-rhyme cliché. Note that `rat`, `fat`, `mat` and `sat` **do rhyme**. The dataset
was never separating rhyming text from non-rhyming text. It was separating one
generator's house style from another's.

### Tier 1 showed the shortcut could not be controlled away

Embedding responses with `all-MiniLM-L6-v2` and fitting a separator on the difference
vectors, then re-fitting on slices where a given surface feature is held equal:

| Slice | Pairs | Accuracy | Drop from raw |
|---|---|---|---|
| raw | 2,000 | 0.989 | — |
| `match_unique_word_ratio` | 1,557 | 0.981 | +0.008 |
| `match_has_comma` | 606 | 0.983 | +0.006 |
| `match_exclamation` | 1,889 | 0.983 | +0.005 |

Controlling the worst confound costs 0.008. The reason is the number underneath:

```
frac_surface_controlled = 0.004        7 pairs out of 2,000
```

In only seven pairs did chosen and rejected share every surface feature. **A
format-controlled evaluation set cannot be constructed from this data, because the
format is the label.** There is no subpopulation large enough to ask whether the model
prefers better rhymes when formatting is held constant.

Then the finding that would have poisoned any downstream policy:

| Measurement | Value |
|---|---|
| cos(prompt, chosen) | 0.469 |
| cos(prompt, rejected) | **0.583** |
| Relevance gap | **−0.115** |
| Rejected responses off-topic | 17.4% |

The rejected response is on average *more* related to its prompt than the chosen one.
A reward model fit here is trained to prefer text that answers the question less
directly. Used as a PPO or GRPO reward, it would actively push the policy off-topic.

### Tier 2 confirmed every prediction

**Experiment A, full finetune.** `distilbert-base-uncased`, mean pooling, prompt and
response visible, 2 epochs on Apple Silicon.

| Result | Value |
|---|---|
| Validation accuracy | **0.99968**, SE 0.00022, on 6,289 held-out-prompt pairs |
| Errors | 2 pairs out of 6,289 |
| Best checkpoint | step 1,000 of 5,458, early stopped at 3,000 |
| Wall clock | 942 s |

Its worst slice out of 22 was `match_newline_count` at 0.998. Holding character length
fixed left it at 0.9987. It did not even need the length shortcut; it read the
distributional difference directly, and that difference survived every control the
slice registry could impose.

**Experiment B, frozen probe.** The same encoder Tier 1 used, frozen, with one linear
layer trained on top and the response-only template, so that the learned weight vector
and Tier 1's principal component live in the same space and their angle means
something.

| Result | Value |
|---|---|
| Validation accuracy | **0.98728**, SE 0.00141 |
| Wall clock | 179 s |
| cos(head weight, Tier-1 PC1) | **+0.735** |
| Worst slice, `match_word_count` | 0.954, a 3.3 point drop from raw |

A frozen off-the-shelf sentence encoder with a single linear layer reached 0.987 in
three minutes. No representation learning was required, because the label direction was
already present in a generic embedding space trained by other people on unrelated data.
The +0.735 cosine is the direct measurement: the axis Tier 1 found with no labels at
all, as the principal component of embedding differences, is the axis the supervised
head learned. The dataset's preference signal and its dominant unsupervised artifact
axis are the same direction.

The capacity-limited probe leaned on length where the full finetune had not, losing 3.3
points when word count was held fixed. Small models reach for the cheapest correlate
available; large ones can afford the richer fingerprint.

### Why the dataset is retired

**No headroom.** Naive Bayes scores 0.998. The finetune scores 0.99968. A transformer,
a Bradley-Terry objective and 942 seconds of training bought 0.0017 over bag of words.
Nothing is left for a reward model to be measurably good or bad at.

**The signal does not exist in the setting the model is for.** A reward model ranks
samples from a policy. Every sample from a rhyming policy is multi-line verse with
commas, so the format tell is constant across the candidates it must rank. On its
actual job distribution the model has no signal.

**The reward points the wrong way.** With a relevance gap of −0.115, optimizing against
this reward moves a policy away from answering its prompt.

### What survived

The instrument. Every finding above came out of this repo's own CLI, and the ledger
predicted the Tier-2 outcomes from Tier-0 and Tier-1 numbers alone. Total cost of the
negative result was about 19 minutes of laptop compute, most of it spent confirming
what the ten-second audit had already said.

The task, too. Rhyming has a property almost no public preference dataset has:
**computable ground truth**, since whether two lines rhyme is decidable by phoneme
lookup with no model involved. That makes it a good controlled testbed for reward
hacking, where the true objective and the learned proxy can be measured separately. The
data needs regenerating from a single policy with programmatic labels, and the audit
thresholds that condemned the old set are written down in advance as the gate the new
one must pass.

---

## The diagnostic stack

| Tier | Cost | Question it answers |
|---|---|---|
| **0** `audit` | Seconds, no model | Can trivial features already predict the label? What is the shortcut? |
| **1** `profile` | Minutes, embeddings | Does the signal survive controlling those shortcuts? Is the data grounded, diverse, leak-free? |
| **2** `eval` + diagnostics | After training | What did the model actually learn? Which direction, which slices, how inflated? |
| **Ledger** `ledger` | Instant | Did Tier 2 confirm what Tiers 0 and 1 predicted? |

The ledger is the piece that makes the rest honest. It writes down a prediction from
cheap analysis, then scores the trained model against it, so a good accuracy number
cannot quietly stand in for understanding. Both runs above ship with theirs.

---

## Install

Requires Python 3.9 or newer. PyTorch with Metal support installs from the default
wheel on Apple Silicon.

```bash
git clone https://github.com/mk-44/rewardlab.git
cd rewardlab
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m rlhf --help
```

All commands run **from the repository root**. Config paths are root-relative.

## Quickstart

```bash
python -m rlhf resplit --config preference_data/data.yaml   # honest prompt-level split
python -m rlhf audit   --config preference_data/data.yaml   # Tier 0
python -m rlhf profile --config preference_data/data.yaml   # Tier 1
python -m rlhf train   --config experiments/reward_model/probe_minilm/config.yaml
python -m rlhf eval    --config experiments/reward_model/probe_minilm/config.yaml
python -m rlhf ledger  --config experiments/reward_model/probe_minilm/config.yaml
```

Every verb takes one YAML config and accepts `--set key=value` overrides.

## Layout

```
rlhf/
  reward_model/
    core/       config, contracts, device, logging
    dataset/    schema, loaders, formats, audit, distribution, splits, collate
    model/      backbone, head, model, losses
    training/   optim, checkpoint, trainer, evaluate, metrics
    analysis/   diagnostics, ledger
    cli/        main
  dpo/ ppo/ grpo/            reserved
tests/                       mirrors the package, 1176 tests
preference_data/             split data, config, Tier 0 and Tier 1 reports
experiments/reward_model/    configs, ledgers and run artifacts per experiment
```

Shared code moves up to `rlhf/core` when a second method actually needs it, not before.

## Design notes

A few decisions that are load-bearing and easy to get wrong:

- **The Bradley-Terry loss is binary cross-entropy on the reward difference**, computed
  through a numerically stable softplus. The naive form overflows at a difference of
  −88.73 in float32 and −18 in float16.
- **The scalar head is pinned to float32** with autocast disabled locally, even when the
  backbone runs in half precision. At a reward magnitude near 100 the float16 spacing
  between representable values is 0.0625, which silently swallows small margin updates.
- **The head carries no bias term.** Bradley-Terry depends only on differences, so a
  bias cancels exactly and its gradient is identically zero.
- **Dropout is disabled by setting `p=0.0`, not by calling `eval()`**, so that chosen and
  rejected in a pair are scored by the same function. Otherwise the comparison is
  between two different models.
- **Weight decay is applied by parameter dimension**, not by name matching. Name
  sniffing for "bias" and "LayerNorm" finds zero parameters on some architectures.
- **Splits are made at the prompt level**, and the audit checks for near-duplicates by
  embedding cosine, since exact-match checks miss paraphrases across the split boundary.

## Status

Reward model training, evaluation and the full diagnostic stack are complete and tested.
DPO, PPO and GRPO are next, with a policy budget of one billion parameters.

The immediate work is moving to a public preference dataset so that results are
reproducible and comparable against published baselines, with the rhyming task retained
as a controlled reward-hacking testbed once its pairs are regenerated.

## License

Not yet chosen.
