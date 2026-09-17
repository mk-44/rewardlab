from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union, Sequence, Optional, TYPE_CHECKING
from pathlib import Path

import torch
from torch import nn
from rlhf.core.config import ConfigError
from rlhf.core.policy.lm import freeze, sequence_logprobs
from rlhf.core.preference.collate import dataset_digest

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from rlhf.core.preference.schema import PairView
    
    from rlhf.dpo.collate import DPOBatch, DPOCollator
    from rlhf.dpo.config import DPOConfig

Reference = Union["LiveReference", "CachedReference"]
LengthNorm = Literal["sum", "mean"]
PathLike = Union[str, Path]
Device = Union[str, torch.device]


@dataclass
class ReferenceReport:
    backend : Literal["live", "cached"] = "live"
    source : str = ""
    fingerprint : str = ""
    data_digest : str = ""
    length_norm : LengthNorm = "sum"
    n_cached : int = 0
    n_lookups : int = 0
    n_forward_passes : int = 0

    def to_dict(self):
        return dict(self.__dict__)


class LiveReference:
    def __init__(
        self,
        model : PreTrainedModel,
        length_norm : LengthNorm = "sum",
        source : str = "",
        autocast_dtype : Optional[torch.dtype] = None
    ):
        self.model = freeze(model)
        self.length_norm = length_norm
        self.autocast_dtype = autocast_dtype
        self.report = ReferenceReport(
            backend = "live",
            source = source or "<clone of policy at init>",
            length_norm = self.length_norm,
        )

    def __call__(self, batch : DPOBatch) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = sequence_logprobs(self.model, batch.input_ids, batch.attention_mask, batch.completion_mask, self.length_norm, autocast_dtype = self.autocast_dtype)
        self.report.n_forward_passes += 1
        self.report.n_lookups += batch.B
        return log_probs[: batch.B], log_probs[batch.B :]


class CachedReference:
    def __init__(
        self,
        path : PathLike,
        fingerprint : str,
        data_digest : str,
        device : Device
    ):
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"reference cache not found: {p}. build it first.")
        blob = torch.load(p, map_location = "cpu", weights_only = True)
        
        blob_fingerprint = str(blob.get("fingerprint", ""))
        if blob_fingerprint != fingerprint:
            raise ConfigError(
                f"reference cache fingerprint mismatch.\n"
                f"    cache : {blob_fingerprint}\n"
                f"    config: {fingerprint}\n"
                f"the cache was built under a different tokenizer, template, max_length, append_eos, length_norm, weights_dtype, compute_dtype or reference checkpoint. "
                f"rebuild it. do not train on it.")
        
        
        blob_digest = str(blob.get("data_digest", ""))
        if not blob_digest:
            raise ConfigError(f"reference cache at {p} predates the dataset-identity check. rebuild it.")
        if blob_digest != data_digest:
            raise ConfigError(
            f"reference cache was built over a different dataset.\n"
            f"    cache : {blob_digest}\n"
            f"    data  : {data_digest}\n"
            f"  the config fingerprint matches ({fingerprint}), so tokenizer, template, "
            f"max_length, append_eos, length_norm, weights_dtype and compute_dtype are all unchanged. the pairs are "
            f"not. uids are positional, so a deleted, inserted or reordered row silently "
            f"re-points every later pair at its neighbour's anchor. rebuild the cache."
        )
        
        uids = blob["uids"]
        
        self._index = {int(u) : i for i, u in enumerate(uids.tolist())}
        self._chosen = blob["chosen"].to(torch.float32)
        self._rejected = blob["rejected"].to(torch.float32)
        self.device = device
        self.report = ReferenceReport(
            backend = "cached", 
            source = str(p), 
            fingerprint = fingerprint,
            data_digest = data_digest,
            length_norm = str(blob.get("length_norm", "sum")), 
            n_cached = len(self._index)
        )

    def __call__(self, batch : DPOBatch) -> tuple[torch.Tensor, torch.Tensor]:
        want : list[int] = batch.uids.tolist()

        try:
            rows = [self._index[int(u)] for u in want]
        except KeyError as e:
            raise ConfigError(
                f"uid {e.args[0]} is not in the reference cache. the cache and "
                f"the dataset disagree. rebuild the cache from the same file the trainer is reading."
            )
        
        idx = torch.tensor(rows, dtype = torch.long)
        device = batch.input_ids.device
        self.report.n_lookups += batch.B
        return self._chosen[idx].to(device), self._rejected[idx].to(device)


def build_reference_cache(
    model : PreTrainedModel,
    collator : DPOCollator,
    pairs : Sequence[PairView],
    path : PathLike,
    fingerprint : str,
    length_norm : LengthNorm = "sum",
    batch_size : int = 16,
    device : Device = "cpu",
    progress_every : int = 0,
    autocast_dtype : Optional[torch.dtype] = None
):
    was_train = model.training
    try:
        model = model.eval().to(device)
        uids, chosen, rejected = [], [], []

        for start in range(0, len(pairs), batch_size):
            b = collator(pairs[start : start + batch_size]).to(device)
            uids.append(b.uids.cpu())

            with torch.inference_mode():
                logps = sequence_logprobs(model, b.input_ids, b.attention_mask, b.completion_mask, length_norm, autocast_dtype = autocast_dtype)
            
            chosen_logps, rejected_logps = logps[: b.B], logps[b.B :]
            chosen.append(chosen_logps.float().cpu())
            rejected.append(rejected_logps.float().cpu())
        
            if progress_every and (start // batch_size) % progress_every == 0:
                print(f"cache in progress, completed : {min(start + batch_size, len(pairs)):,}/{len(pairs):,}")
            
        
        blob = {
            "fingerprint" : fingerprint,
            "data_digest" : dataset_digest(pairs),
            "length_norm" : length_norm,
            "uids" : torch.cat(uids),
            "chosen" : torch.cat(chosen),
            "rejected" : torch.cat(rejected),
        }

        p = Path(path)
        p.parent.mkdir(parents = True, exist_ok = True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(blob, tmp)
        tmp.replace(p)
        return blob
    finally:
        model.train(was_train)


def make_reference(
    cfg : DPOConfig,
    policy_model : PreTrainedModel,
    collator : Optional[DPOCollator] = None,
    pairs : Optional[Sequence[PairView]] = None,
    device : Device = "cpu",
    train : bool = True,
    autocast_dtype : Optional[torch.dtype] = None
):
    ref_cache_path = cfg.reference.precomputed_logp_path if train else cfg.reference.precomputed_val_logp_path
    fp = cfg.ref_fingerprint()

    if ref_cache_path is not None and Path(ref_cache_path).exists():
        if pairs is None:
            raise ConfigError(f"pairs must be passed for dataset order verification.")
        return CachedReference(ref_cache_path, fp, dataset_digest(pairs), device)
    
    ref_ckpt = cfg.reference.reference_ckpt
    if ref_ckpt:
        from rlhf.core.policy.lm import load_policy
        ref_model, _ = load_policy(model_ckpt = ref_ckpt, device = device, weights_dtype = cfg.execution.weights_dtype)
    else:
        import copy
        ref_model = copy.deepcopy(policy_model)

    if not ref_cache_path:
        return LiveReference(ref_model, cfg.loss.length_norm, source = ref_ckpt or "<clone of policy at init>", autocast_dtype = autocast_dtype)
    else:
        if collator is None or pairs is None:
            raise ConfigError(f"reference cache {ref_cache_path} does not exist and no collator/pairs were given to build it")
        blob = build_reference_cache(ref_model, collator, pairs, ref_cache_path, fp, cfg.loss.length_norm, batch_size = cfg.train.eval_batch_size, device = device, autocast_dtype = autocast_dtype)
        return CachedReference(ref_cache_path, fp, blob["data_digest"], device)


def assert_step_zero(
    reference : Reference,
    policy_model : PreTrainedModel,
    batch : DPOBatch,
    length_norm : LengthNorm = "sum",
    tol : float = 0.0,
    autocast_dtype : Optional[torch.dtype] = None
) -> tuple(float, float):
    with torch.no_grad():
        policy_logps = sequence_logprobs(policy_model, batch.input_ids, batch.attention_mask, batch.completion_mask, length_norm, autocast_dtype = autocast_dtype)
    pc, pr = policy_logps[: batch.B], policy_logps[batch.B :]
    rc, rr = reference(batch)

    chosen = pc - rc
    rejected = pr - rr
    worst_chosen = chosen.abs().max()
    worst_rejected = rejected.abs().max()

    if worst_chosen > tol:
        raise ConfigError(
            f"step-0 identity violated: max |chosen_log_probs - rejected_log_probs| = {worst_chosen:.3e}, expected <= {tol}.\n"
            f"    pi_ref is supposed to be a frozen clone of pi_theta, so every "
            f"log-ratio should be exactly 0.\n"
            f"    check: length_norm on both sides, device, weights_dtype, compute_dtype, and whether "
            f"the cache was built with this exact collator.")
    
    if worst_rejected > tol:
        raise ConfigError(
            f"step-0 identity violated: max |chosen_log_probs - rejected_log_probs| = {worst_rejected:.3e}, expected <= {tol}.\n"
            f"    pi_ref is supposed to be a frozen clone of pi_theta, so every "
            f"log-ratio should be exactly 0.\n"
            f"    check: length_norm on both sides, device, weights_dtype, compute_dtype, and whether "
            f"the cache was built with this exact collator.")
    return (worst_chosen, worst_rejected)


def render(rep : ReferenceReport, width : int = 76) -> str:
    bar = "=" * width
    L = [bar, "REFERENCE", bar,
         f"  backend       : {rep.backend}",
         f"  source        : {rep.source}",
         f"  length_norm   : {rep.length_norm}",
         f"  lookups       : {rep.n_lookups:,}"]
    if rep.backend == "cached":
        L.append(f"  cached pairs  : {rep.n_cached:,}")
        L.append(f"  fingerprint   : {rep.fingerprint}")
        L.append(f"  data_digest   : {rep.data_digest}")
    else:
        L.append(f"  forward passes: {rep.n_forward_passes:,}   <- the cost the cache removes")
    L.append(bar)
    return "\n".join(L)
