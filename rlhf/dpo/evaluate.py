from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Iterable, Union, Tuple, Sequence, TYPE_CHECKING
import torch
import time
from rlhf.core.config import ConfigError
from rlhf.core.losses import bt_loss
from rlhf.core.policy.lm import generate, sequence_logprobs
from rlhf.core.scoring.rewards import CustomRewardFunction, RewardReport, combine_rewards

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from rlhf.dpo.collate import DPOBatch
    from rlhf.dpo.config import DPOConfig
    from rlhf.dpo.reference import CachedReference, LiveReference


Reference = Union["LiveReference", "CachedReference"]
LengthNorm = Literal["sum", "mean"]
Device = Union[str, torch.device]


@dataclass
class PreferenceMetrics:
    n_pairs : int = 0
    loss : float = 0.0
    accuracy : float = 0.0

    delta_mean : float = 0.0
    reward_chosen : float = 0.0
    reward_rejected : float = 0.0
    reward_margin : float = 0.0

    policy_logp_chosen : float = 0.0
    policy_logp_rejected : float = 0.0
    ref_logp_chosen : float = 0.0
    ref_logp_rejected : float = 0.0

    logratio_chosen : float = 0.0
    logratio_rejected : float = 0.0

    frac_winner_below_ref : float = 0.0
    frac_loser_below_ref : float = 0.0

    frac_pairs_saturated : float = 0.0
    grad_weight_mean : float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class GenerationMetrics:
    n_prompts : int = 0
    n_samples : int = 0
    mean_new_tokens : float = 0.0
    frac_empty : float = 0.0

    reward_mean : Optional[float] = None
    reward_min : Optional[float] = None
    reward_max : Optional[float] = None
    reward : Optional[RewardReport] = None
    reason : Optional[str] = None

    examples : list = field(default_factory = list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["reward"] = self.reward.to_dict() if self.reward is not None else None
        return d


@dataclass
class EvalReport:
    preference : PreferenceMetrics = field(default_factory = PreferenceMetrics)
    slices : dict = field(default_factory = dict)
    generation : Optional[GenerationMetrics] = None
    n_batches : int = 0
    seconds : float = 0.0

    def to_dict(self) -> dict:
        return {
            "preference" : self.preference.to_dict(),
            "slices" : {k : v.to_dict() for k, v in self.slices.items()},
            "generation" : self.generation.to_dict() if self.generation else None,
            "n_batches" : self.n_batches,
            "seconds" : self.seconds,
        }


def preference_metrics(
    policy_chosen_logps : torch.Tensor,
    policy_rejected_logps : torch.Tensor,
    ref_chosen_logps : torch.Tensor,
    ref_rejected_logps : torch.Tensor,
    beta : float = 0.1,
) -> PreferenceMetrics:

    if beta <= 0.0:
        raise ConfigError(f"beta must be > 0, given beta = {beta}")
    shapes = [x.shape for x in (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)]
    if any(len(s) != 1 for s in shapes) or len(set(shapes)) != 1:
        raise ConfigError(f"all four inputs / log_probs must be [B] and the same length, got {shapes}")

    pc, pr = policy_chosen_logps.float(), policy_rejected_logps.float()
    rc, rr = ref_chosen_logps.float(), ref_rejected_logps.float()
    r_chosen = beta * (pc - rc)
    r_rejected = beta * (pr - rr)
    per_pair_loss = bt_loss(r_chosen, r_rejected, reduction = "none")
    delta = r_chosen - r_rejected
    grad_wt = torch.sigmoid(-delta)
    logratio_chosen = pc - rc
    logratio_rejected = pr - rr
    n_pairs = len(policy_chosen_logps)

    return PreferenceMetrics(
        n_pairs = n_pairs,
        loss = float(per_pair_loss.mean()),
        accuracy = float((delta > 0).float().mean()),
        delta_mean = float(delta.mean()),
        reward_chosen = float((r_chosen).mean()),
        reward_rejected = float((r_rejected).mean()),
        reward_margin = float((r_chosen - r_rejected).mean()),
        policy_logp_chosen = float(pc.mean()),
        policy_logp_rejected = float(pr.mean()),
        ref_logp_chosen = float(rc.mean()),
        ref_logp_rejected = float(rr.mean()),
        logratio_chosen = float(logratio_chosen.mean()),
        logratio_rejected = float(logratio_rejected.mean()),
        frac_winner_below_ref = float((pc < rc).float().mean()),
        frac_loser_below_ref = float((pr < rr).float().mean()),
        frac_pairs_saturated = float((grad_wt < 0.01).float().mean()),
        grad_weight_mean = float(grad_wt.mean())
    )


def slice_metrics(
    policy_chosen_logps : torch.Tensor,
    policy_rejected_logps : torch.Tensor,
    ref_chosen_logps : torch.Tensor,
    ref_rejected_logps : torch.Tensor,
    slices : Sequence[dict],
    beta : float = 0.1,
) -> dict:
    n = len(policy_chosen_logps)
    if len(slices) != n:
        raise ConfigError(f"len(slices) {len(slices)} must be = len(policy_chosen_logps) {len(policy_chosen_logps)}")
    
    grps = {}
    for i, dct in enumerate(slices):
        for k, v in dct.items():
            if not v:
                continue
            grps.setdefault(f"{k}={v}", []).append(i)
    
    res = {}
    for lbl in sorted(grps):
        idx = torch.tensor(grps[lbl], dtype = torch.long)
        res[lbl] = preference_metrics(
            policy_chosen_logps[idx], 
            policy_rejected_logps[idx],
            ref_chosen_logps[idx], 
            ref_rejected_logps[idx], 
            beta = beta,
        )
    
    return res


def evaluate_preferences(
    policy_model : PreTrainedModel,
    reference : Reference,
    batches : Iterable[DPOBatch],
    beta : float = 0.1,
    length_norm : LengthNorm = "sum",
    device : Device = "cpu",
    with_slices : bool = True,
) -> EvalReport:
    t_start = time.perf_counter()
    was_train = policy_model.training
    pcs, prs, rcs, rrs = [], [], [], []
    all_slices, n_batches = [], 0

    try:
        policy_model.eval()
        with torch.no_grad():
            for batch in batches:
                b = batch.to(device)
                logps = sequence_logprobs(policy_model, b.input_ids, b.attention_mask, b.completion_mask, length_norm)
                pcs.append(logps[: b.B].float().detach().cpu())
                prs.append(logps[b.B :].float().detach().cpu())

                rc_batch, rr_batch = reference(b)
                rcs.append(rc_batch.float().cpu())
                rrs.append(rr_batch.float().cpu())

                all_slices.extend(b.slices)
                n_batches += 1
    finally:
        policy_model.train(was_train)
    
    if n_batches == 0:
        raise ConfigError("evaluate_preferences got zero batches")
    
    pc = torch.cat(pcs, dim = 0)
    pr = torch.cat(prs, dim = 0)
    rc = torch.cat(rcs, dim = 0)
    rr = torch.cat(rrs, dim = 0)

    return EvalReport(
        preference = preference_metrics(pc, pr, rc, rr, beta),
        slices = slice_metrics(pc, pr, rc, rr, all_slices, beta) if with_slices and all_slices else {},
        n_batches = n_batches,
        seconds = time.perf_counter() - t_start    
    )


def evaluate_generations(
    policy_model : PreTrainedModel,
    tokenizer : PreTrainedTokenizerBase,
    prompts : Sequence[str],
    reward_functions : Optional[Sequence[CustomRewardFunction]] = None,
    reward_wts : Optional[Sequence[float]] = None,
    normalize_reward_func_wts : bool = False,
    num_samples_per_prompt : int = 1,
    temperature : float = 1.0,
    top_p : float = 0.95,
    max_new_tokens : int = 32,
    batch_size : int = 8,
    seed : Optional[int] = None,
    n_examples : int = 4,
) -> GenerationMetrics:
    if len(prompts) < 1:
        raise ConfigError(f"Pass atleats 1 prompt, currently {len(prompts)} prompts passed")
    
    if batch_size < 1:
        raise ConfigError(f"batch_size must be > 1, currently its {batch_size}")
    
    if num_samples_per_prompt < 1:
        raise ConfigError(f"num_samples_per_prompt must be >= 1, got {num_samples_per_prompt}")
    
    
    was_train = policy_model.training
    prompts_flat, responses_flat = [], []
    rng_state = torch.get_rng_state() if seed is not None else None

    try:
        policy_model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i : i + batch_size])
            out = generate(policy_model, tokenizer, batch, num_samples_per_prompt, temperature, top_p, max_new_tokens, seed)
            for pr, res_list in zip(batch, out):
                for res in res_list:
                    prompts_flat.append(pr)
                    responses_flat.append(res)
    finally:
        policy_model.train(was_train)
        if rng_state is not None:
            torch.set_rng_state(rng_state)
    
    len_responses_ = [len(tokenizer(res, add_special_tokens = False)["input_ids"]) for res in responses_flat]
    total_rew, rew_report = combine_rewards(prompts_flat, responses_flat, reward_functions, reward_wts, normalize_reward_func_wts)
    examples = [(prompts_flat[i], responses_flat[i], None if total_rew is None else total_rew[i]) for i in range(min(n_examples, len(responses_flat)))]
    num_responses = len(responses_flat)

    return GenerationMetrics(
        n_prompts = len(prompts),
        n_samples = num_responses,
        mean_new_tokens = sum(len_responses_) / num_responses if num_responses > 0 else 0.0,
        frac_empty = sum(1 for r in responses_flat if not r.strip()) / num_responses if num_responses > 0 else 0.0,
        reward_mean = sum(total_rew) / len(total_rew) if total_rew is not None else None,
        reward_max = max(total_rew) if total_rew is not None else None,
        reward_min = min(total_rew) if total_rew is not None else None,
        reward = rew_report,
        reason = rew_report.reason,
        examples = examples
    )


def evaluate(
    cfg : DPOConfig,
    policy_model : PreTrainedModel,
    reference : Reference,
    batches : Iterable[DPOBatch],
    tokenizer : Optional[PreTrainedTokenizerBase] = None,
    prompts : Optional[Sequence[str]] = None,
    reward_functions : Optional[Sequence[CustomRewardFunction]] = None,
    device : Device = "cpu",
    with_slices : bool = True,
    seed : Optional[int] = None,
) -> EvalReport:
    report = evaluate_preferences(
        policy_model, 
        reference, 
        batches,
        beta = cfg.loss.beta,
        length_norm = cfg.loss.length_norm,
        device = device,
        with_slices = with_slices
    )

    if tokenizer is not None and prompts:
        inf_config = cfg.inference
        report.generation = evaluate_generations(
            policy_model, 
            tokenizer, 
            prompts,
            reward_functions = reward_functions,
            reward_wts = cfg.reward.reward_wts or None,
            normalize_reward_func_wts = cfg.reward.normalize_reward_func_wts,
            num_samples_per_prompt = inf_config.num_samples_per_prompt,
            temperature = inf_config.temperature,
            top_p = inf_config.top_p,
            max_new_tokens = inf_config.max_new_tokens,
            seed = seed
        )
    return report


def render(rep : EvalReport, width : int = 76, max_slices : int = 12) -> str:
    bar = "=" * width
    p = rep.preference
    L = [bar, "DPO EVAL", bar,
         f"  pairs         : {p.n_pairs:,}   in {rep.n_batches} batches   {rep.seconds:.1f}s",
         f"  loss          : {p.loss:8.4f}   accuracy {p.accuracy:.3f}",
         f"  Delta         : {p.delta_mean:+8.4f}   saturated {p.frac_pairs_saturated:.1%}",
         f"  reward        : chosen {p.reward_chosen:+7.4f}   rejected {p.reward_rejected:+7.4f}"
         f"   margin {p.reward_margin:+7.4f}",
         "",
         f"  policy log pi : chosen {p.policy_logp_chosen:9.3f}   rejected {p.policy_logp_rejected:9.3f}",
         f"  ref    log pi : chosen {p.ref_logp_chosen:9.3f}   rejected {p.ref_logp_rejected:9.3f}",
         f"  log ratio     : chosen {p.logratio_chosen:+9.3f}   rejected {p.logratio_rejected:+9.3f}",
         f"  BELOW REF     : winner {p.frac_winner_below_ref:.1%}   loser {p.frac_loser_below_ref:.1%}",
         ]
    if p.frac_winner_below_ref > 0.5:
        L.append("                  the winner's likelihood is falling on most pairs.")
        L.append("                  Delta can still rise while the model gets worse.")

    if rep.slices:
        L += ["", f"  {'slice':<28}{'n':>7}{'acc':>8}{'Delta':>10}{'win<ref':>10}"]
        for lbl in list(rep.slices)[:max_slices]:
            s = rep.slices[lbl]
            L.append(f"  {lbl[:28]:<28}{s.n_pairs:>7}{s.accuracy:>8.3f}"
                     f"{s.delta_mean:>+10.4f}{s.frac_winner_below_ref:>10.1%}")
        if len(rep.slices) > max_slices:
            L.append(f"  ... {len(rep.slices) - max_slices} more")

    g = rep.generation
    if g is not None:
        L += ["", f"  generations   : {g.n_samples:,} from {g.n_prompts:,} prompts"
                  f"   {g.mean_new_tokens:.1f} new tokens   empty {g.frac_empty:.1%}"]
        if g.reward_mean is None:
            L.append(f"  reward        : not computed ({g.reason})")
        else:
            L.append(f"  reward        : mean {g.reward_mean:+8.4f}"
                     f"   min {g.reward_min:+8.4f}   max {g.reward_max:+8.4f}")
            for nm, w, mu in zip(g.reward.names, g.reward.weights, g.reward.means):
                L.append(f"     {nm[:22]:<22} w={w:<8.4f} mean {mu:+8.4f}")
        for prompt, response, score in g.examples:
            tag = "" if score is None else f"  [{score:+.3f}]"
            L.append(f"     {prompt[:30]!r} -> {response[:34]!r}{tag}")

    L.append(bar)
    return "\n".join(L)
