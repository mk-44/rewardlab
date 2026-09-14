# rewardlab

A from scratch RLHF stack written in plain PyTorch. It trains reward models on preference data and then trains a policy with direct preference optimisation (DPO). PPO and GRPO come next.

## Purpose

The point of this code is to be read. Every numerical decision that a training framework usually hides is written out and tested here: how a preference pair becomes a loss and how a sequence log probability is computed and how the reference model is frozen and cached and what gets logged at every step.

Three rules shape the package.

* Everything a run needs is declared in one yaml. Nothing is inferred from the data.
* Nothing task specific lives inside `rlhf/`. A project supplies its own reward functions in its own files and names them in the yaml.
* Every run leaves receipts on disk: config stamps and metrics and checkpoints and an eval report.

## Important files

* `rlhf/__main__.py` routes `rlhf <method> <verb>` to the method's command line.
* `rlhf/core/config.py` loads a yaml into dataclasses and applies key=value overrides. Unknown keys are refused.
* `rlhf/core/preference/loaders.py` and `rlhf/core/preference/schema.py` read pairwise and k way preference files into groups and pairs.
* `rlhf/core/scoring/rewards.py` loads reward functions from `path/to/file.py:function` specs and combines them with weights.
* `rlhf/core/training/checkpoint.py` saves and restores model plus optimizer plus scheduler plus RNG state.
* `rlhf/core/logging.py` is the RunLogger. It claims the run directory and writes `metrics.jsonl` and `console.log` and json stamps.
* `rlhf/core/device.py` resolves device and dtype into an ExecutionPlan and refuses what the machine cannot do.
* `rlhf/core/policy/lm.py` loads a causal language model and generates from it.
* `rlhf/reward/model/model.py` is the reward model: a backbone with pooling and a scalar head.
* `rlhf/reward/training/trainer.py` trains it on the Bradley Terry loss.
* `rlhf/reward/scoring.py` turns a finished reward run into a reward function through `from_run`.
* `rlhf/dpo/collate.py` tokenises a pair under the template and builds the completion mask.
* `rlhf/dpo/losses.py` holds the sequence log probability and the DPO loss.
* `rlhf/dpo/reference.py` builds the frozen reference and its precomputed log probability cache.
* `rlhf/dpo/evaluate.py` scores a checkpoint on the validation pairs and on generated samples.
* `rlhf/dpo/trainer.py` is the DPO training loop with checkpoint selection and early stopping.
* `rlhf/dpo/cli/main.py` exposes the verbs for cache building and training and eval and export.
* `experiments/dpo/non_sft_50k_data/` holds one experiment: `config.yaml` and `custom_reward_functions.py` and `run.sh` and its own README with the exact steps.
* `data/preference_data/rhyme_50k/` holds the preference splits used by that experiment.

## Formulas

### Reward model

A pair says that the chosen response `y_w` beats the rejected response `y_l` for prompt `x`. It does not say that `y_w` is a 1 and `y_l` is a 0. So the loss compares the two rewards instead of scoring them one at a time.

```
p(y_w beats y_l | x) = e^{r_w} / (e^{r_w} + e^{r_l}) = σ(r_w − r_l)

loss = −log σ(r_w − r_l) = softplus(r_l − r_w)

∂loss/∂θ = −σ(−Δ) · ∂Δ/∂θ      with Δ = r_w − r_l
```

The more wrong the pair the larger the gradient. Adding a constant to every reward leaves the loss unchanged. Per response binary cross entropy would instead push `r_w` toward plus infinity and `r_l` toward minus infinity and that is not what the data says.

### From a reward to a policy

Maximising expected reward alone collapses the policy onto the single best response. A KL penalty toward a reference policy keeps it honest.

```
max over π of  E_{y ~ π(·|x)} r(y|x)  −  β · KL(π(·|x) || π_ref(·|x))
```

Setting the derivative of the Lagrangian to zero gives the closed form.

```
r(y|x) − β · (log π(y|x)/π_ref(y|x) + 1) − λ = 0

π*(y|x) = π_ref(y|x) · e^{r(y|x)/β} / Z(x)       Z(x) = Σ_y π_ref(y|x) · e^{r(y|x)/β}

r(y|x) = β · log π*(y|x)/π_ref(y|x) + β · log Z(x)
```

### DPO

Substituting that reward into the Bradley Terry loss cancels `Z(x)` because both responses share the prompt.

```
s_w = log π_θ(y_w|x) / π_ref(y_w|x)
s_l = log π_θ(y_l|x) / π_ref(y_l|x)
Δ   = s_w − s_l

loss = −log σ(β · Δ) = softplus(−β · Δ)

∇_θ loss = −β · σ(−β · Δ) · ∇_θ Δ
```

The implicit reward is `β · log π_θ(y|x)/π_ref(y|x)` and no explicit reward model is needed for training. When `Δ` is far below zero the weight `σ(−β · Δ)` is near one and the update is large. When `Δ` is far above zero the weight is near zero. At step zero the policy equals the reference so `Δ = 0` and the loss is `ln 2 = 0.693`. An optional SFT term adds `sft_wt` times the negative per token mean of `log π_θ(y_w|x)`.

### Sequence log probability

```
log π(y|x) = Σ_t log π(y_t | x y_<t)     summed over completion tokens only

log softmax:  log (e^{z_i} / Σ_j e^{z_j}) = z_i − logsumexp(z) = (z_i − m) − log Σ_j e^{z_j − m}     m = max_j z_j
```

The policy forward pass returns logits for every position. The logits are shifted one position back against the input ids and the log probability of each actual next token is gathered and multiplied by the shifted completion mask and summed. The reference model runs under no gradient or its values come from the precomputed cache. `−log σ(β · Δ)` is computed through `logsigmoid` for numerical stability.

## Running the experiment

The steps are in `experiments/dpo/non_sft_50k_data/README.md`. The whole pipeline is one script.

```
sh experiments/dpo/non_sft_50k_data/run.sh
```

## License

MIT
