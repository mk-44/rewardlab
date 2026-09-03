from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional
import torch
from rlhf.reward_model.core.contracts import ConfigError
from rlhf.reward_model.dataset.schema import PairView
from transformers import AutoTokenizer
import numpy as np


_UID_GROUP = 1_000_000
_UID_IDX = 1_000


def pair_uid(p : PairView):
    if not((0 <= p.chosen_idx < _UID_IDX) and (0 <= p.rejected_idx < _UID_IDX)):
        raise ConfigError(f"pair indices ({p.chosen_idx}, {p.rejected_idx}) exceed uid capacity {_UID_IDX}; raise _UID_IDX if you truly have K >= {_UID_IDX}")
    return (p.group_uid * _UID_GROUP) + (p.chosen_idx * _UID_IDX) + (p.rejected_idx)


@dataclass
class PairBatch:
    chosen_input_ids : torch.Tensor
    chosen_mask : torch.Tensor
    rejected_input_ids : torch.Tensor
    rejected_mask : torch.Tensor
    uids : torch.Tensor
    slices : list = field(default_factory = list)
    
    def __len__(self) -> int:
        return self.chosen_input_ids.shape[0]

    def to(self, device) -> PairBatch:
        return PairBatch(
            chosen_input_ids = self.chosen_input_ids.to(device),
            chosen_mask= self.chosen_mask.to(device),
            rejected_input_ids = self.rejected_input_ids.to(device),
            rejected_mask = self.rejected_mask.to(device),
            uids = self.uids.to(device),
            slices = self.slices
        )
    
    def joined(self, pad_id : int):
        max_L = max(self.chosen_input_ids.shape[1], self.rejected_input_ids.shape[1])

        def pad_to(x : torch.Tensor, value : int) -> torch.Tensor:
            if x.shape[1] == max_L:
                return x
            extra = torch.full((x.shape[0], max_L - x.shape[1]), value, dtype = x.dtype, device = x.device)
            return torch.cat([x, extra], dim = 1)
        
        ids = torch.cat([pad_to(self.chosen_input_ids, pad_id), pad_to(self.rejected_input_ids, pad_id)], dim = 0)
        mask = torch.cat([pad_to(self.chosen_mask, 0), pad_to(self.rejected_mask, 0)], dim = 0)
        return ids, mask


@dataclass
class CollateReport:
    tokenizer : str = ""
    template : str = ""
    max_length : int = 0
    pad_token_added : bool = False
    n_pairs : int = 0
    n_truncated_chosen : int = 0
    n_truncated_rejected : int = 0
    max_len_seen_chosen : int = 0
    max_len_seen_rejected : int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class PairCollator:
    def __init__(self, tokenizer_name : str, max_length : int = 512, template : str = "{prompt}\n{response}", tokenizer : Optional[AutoTokenizer] = None):
        if "{response}" not in template:
            raise ConfigError(f"template must contain {{response}}, got {template!r}")
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.tok = tokenizer
        self.max_length = max_length
        self.template = template

        pad_added = False
        if self.tok.pad_token is None:
            if self.tok.eos_token is None:
                raise ConfigError(f"tokenizer {tokenizer_name!r} has neither pad nor eos token, declare a tokenizer that can pad")
            self.tok.pad_token = self.tok.eos_token
            pad_added = True

        self.report = CollateReport(tokenizer = tokenizer_name, template = template, max_length = max_length, pad_token_added = pad_added)
    
    def format_pair(self, prompt : str, response : str) -> str:
        return self.template.format(prompt = prompt, response = response)

    def _encode(self, texts : list) -> tuple:
        enc = self.tok(texts, padding = True, truncation = True, max_length = self.max_length, return_tensors = "pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        lengths = mask.sum(dim = 1)
        n_trunc = 0
        for i in (lengths == self.max_length).nonzero().flatten().tolist():
            full = self.tok(texts[i], truncation=False)["input_ids"]
            if len(full) > self.max_length:
                n_trunc += 1
        return ids, mask, n_trunc
    
    def __call__(self, pairs : Sequence[PairView]) -> PairBatch:
        if len(pairs) == 0:
            raise ConfigError("cannot collate an empty list of pairs")

        chosen_texts = [self.format_pair(p.prompt, p.chosen) for p in pairs]
        rejected_texts = [self.format_pair(p.prompt, p.rejected) for p in pairs]
        c_ids, c_mask, c_trunc = self._encode(chosen_texts)
        r_ids, r_mask, r_trunc = self._encode(rejected_texts)

        rep = self.report
        rep.n_pairs += len(pairs)
        rep.n_truncated_chosen += c_trunc
        rep.n_truncated_rejected += r_trunc
        rep.max_len_seen_chosen = max(rep.max_len_seen_chosen, int(c_mask.sum(1).max()))
        rep.max_len_seen_rejected = max(rep.max_len_seen_rejected, int(r_mask.sum(1).max()))

        return PairBatch(
            chosen_input_ids = c_ids, 
            chosen_mask = c_mask,
            rejected_input_ids = r_ids, 
            rejected_mask = r_mask,
            uids = torch.tensor([pair_uid(p) for p in pairs], dtype = torch.int64),
            slices = [dict(p.slices) for p in pairs]
        )

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id


def render(rep: CollateReport, width: int = 76) -> str:
    bar = "=" * width
    trunc_c = rep.n_truncated_chosen / max(rep.n_pairs, 1)
    trunc_r = rep.n_truncated_rejected / max(rep.n_pairs, 1)
    L = [bar, "COLLATE REPORT", bar,
         f"  tokenizer       : {rep.tokenizer}"
         + ("   (pad token added = eos)" if rep.pad_token_added else ""),
         f"  template        : {rep.template!r}",
         f"  max_length      : {rep.max_length}",
         f"  pairs seen      : {rep.n_pairs:,}",
         f"  truncated       : chosen {rep.n_truncated_chosen:,} ({trunc_c:.1%})"
         f"   rejected {rep.n_truncated_rejected:,} ({trunc_r:.1%})",
         f"  longest seen    : chosen {rep.max_len_seen_chosen}"
         f"   rejected {rep.max_len_seen_rejected}",
         bar]
    if abs(trunc_c - trunc_r) > 0.05:
        L.insert(-1, f"  !  ASYMMETRIC TRUNCATION  one side loses text "
                     f"{abs(trunc_c - trunc_r):.1%} more often — the comparison "
                     f"is no longer between what the annotator saw (F1).")
    return "\n".join(L)


class PretokenizedPairs:
    def __init__(self, pairs : Sequence[PairView], collator : PairCollator):
        self.collator = collator
        self.pairs = list(pairs)
        self.report = CollateReport(
            tokenizer = collator.report.tokenizer,
            template = collator.report.template,
            max_length = collator.report.max_length,
            pad_token_added = collator.report.pad_token_added
        )
        _dtype = np.uint16 if len(collator.tok) < 65_536 else np.int32
        self._dtype = _dtype
        sides = {}
        for side, texts in [
            ("chosen", [collator.format_pair(p.prompt, p.chosen) for p in pairs]), 
            ("rejected", [collator.format_pair(p.prompt, p.rejected) for p in pairs])
        ]:
            ids, mask, n_trunc = collator._encode(texts)

            max_id = int(ids.max())  
            if max_id > np.iinfo(_dtype).max or int(ids.min()) < 0:
                raise ConfigError(
                    f"token id {max_id} exceeds {_dtype.__name__} capacity {np.iinfo(_dtype).max}; tokenizer"
                    f"{collator.report.tokenizer!r} emits ids outside [0, len(tokenizer))"                                   
                )      

            lengths = mask.sum(dim = 1).tolist()
            flat = np.empty(int(sum(lengths)), dtype = _dtype)
            offsets = np.zeros(len(texts) + 1, dtype = np.int64)
            pos = 0

            for i, (row, n) in enumerate(zip(ids.tolist(), lengths)):
                flat[pos : pos + n] = row[: n]
                pos += n
                offsets[i + 1] = pos
            
            sides[side] = (flat, offsets)
            
            if side == "chosen":
                self.report.n_truncated_chosen = n_trunc
                self.report.max_len_seen_chosen = int(max(lengths))
            else:
                self.report.n_truncated_rejected = n_trunc
                self.report.max_len_seen_rejected = int(max(lengths))
        
        self.report.n_pairs = len(self.pairs)
        self._chosen, self._rejected = sides["chosen"], sides["rejected"]
        self._uids = torch.tensor([pair_uid(p) for p in self.pairs], dtype = torch.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def n_bytes(self) -> int:
        return int(self._chosen[0].nbytes + self._rejected[0].nbytes + self._chosen[1].nbytes + self._rejected[1].nbytes)
    
    def _gather(self, store, indices : Sequence[int], pad_id : int):
        flat, offsets = store
        lengths = [offsets[i + 1] - offsets[i] for i in indices]
        L = max(lengths)
        ids = torch.full((len(indices), L), pad_id, dtype = torch.int64)
        mask = torch.zeros((len(indices), L), dtype = torch.int64)

        for row, (i, n) in enumerate(zip(indices, lengths)):
            chunk = flat[int(offsets[i]): int(offsets[i]) + n].astype("int64")
            ids[row, :n] = torch.from_numpy(chunk)
            mask[row, :n] = 1
        return ids, mask
    
    def batch(self, indices: Sequence[int]) -> PairBatch:
        if len(indices) == 0:
            raise ConfigError("cannot build a batch from an empty index list")

        pad = self.collator.pad_id
        c_ids, c_mask = self._gather(self._chosen, indices, pad)
        r_ids, r_mask = self._gather(self._rejected, indices, pad)
        
        return PairBatch(
            chosen_input_ids = c_ids, 
            chosen_mask = c_mask,
            rejected_input_ids = r_ids, 
            rejected_mask = r_mask,
            uids = self._uids[list(indices)],
            slices = [dict(self.pairs[i].slices) for i in indices]
        )
