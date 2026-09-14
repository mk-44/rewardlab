from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import torch
from rlhf.core.contracts import ConfigError
from rlhf.core.preference.collate import pair_uid
from rlhf.core.preference.schema import PairView


@dataclass
class DPOBatch:
    input_ids : torch.Tensor
    attention_mask : torch.Tensor
    completion_mask : torch.Tensor
    B : int
    uids : torch.Tensor
    slices : list = field(default_factory = list)

    def __len__(self):
        return self.B
    
    def to(self, device : str):
        return DPOBatch(
            input_ids = self.input_ids.to(device),
            attention_mask = self.attention_mask.to(device),
            completion_mask = self.completion_mask.to(device),
            B = self.B,
            uids = self.uids.to(device),
            slices = self.slices
        )


@dataclass
class DPOCollateReport:
    tokenizer : str = ""
    template : str = ""
    max_length : int = 0
    append_eos : bool = True
    pad_token_added : bool = False
    padding_side : str = "right"
    n_pairs : int = 0
    n_prompt_truncated : int = 0
    n_response_truncated : int = 0
    max_len_seen : int = 0
    min_completion_len : int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class DPOCollator:
    def __init__(
        self,
        tokenizer_name : str,
        max_length : int = 512,
        template : str = "{prompt}\n{response}",
        append_eos : bool = True,
        tokenizer  = None
    ):
        for ph in ("{prompt}", "{response}"):
            if ph not in template:
                raise ConfigError(f"template must contain {ph}, got {template!r}")

        if template.index("{prompt}") > template.index("{response}"):
            raise ConfigError(f"template must put {{prompt}} before {{response}}, got {template!r}")
        
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.tok = tokenizer
        self.tok.padding_side = "right"
        pad_added = False

        if self.tok.pad_token is None:
            if self.tok.eos_token is None:
                raise ConfigError(
                    f"tokenizer {tokenizer_name!r} has neither pad nor eos token")
            self.tok.pad_token = self.tok.eos_token
            pad_added = True
        if append_eos and self.tok.eos_token_id is None:
            raise ConfigError(f"append_eos=True but tokenizer {tokenizer_name!r} has no eos token")
        
        self.max_length = max_length
        self.template = template
        self.append_eos = append_eos

        head, tail = template.split("{response}")
        self.head_tpl = head
        self.tail_tpl = tail

        self.report = DPOCollateReport(
            tokenizer = tokenizer_name, 
            template = template, 
            max_length = max_length,
            append_eos = append_eos, 
            pad_token_added = pad_added, 
            padding_side ="right"
        )


    def encode_one(self, prompt : str, response : str) -> tuple:
        head_text = self.head_tpl.format(prompt = prompt)
        prompt_ids = self.tok(head_text, add_special_tokens = False)["input_ids"]
        resp_ids = self.tok(response + self.tail_tpl, add_special_tokens = False)["input_ids"]

        if self.append_eos:
            resp_ids = resp_ids + [self.tok.eos_token_id]
        
        p_trunc = r_trunc = False
        budget = self.max_length - len(resp_ids)

        if budget < 0:
            prompt_ids = []
            resp_ids = resp_ids[: self.max_length]
            r_trunc = True
        elif len(prompt_ids) > budget:
            prompt_ids = prompt_ids[len(prompt_ids) - budget :]
            p_trunc = True
        
        ids = prompt_ids + resp_ids
        if ids[ : len(prompt_ids)] != prompt_ids:
            raise ConfigError("completion boundary drifted: ids[:P] != prompt_ids")
        return ids, len(prompt_ids), p_trunc, r_trunc

    def __call__(self, pairs : Sequence[PairView]) -> DPOBatch:
        if len(pairs) == 0:
            raise ConfigError("cannot collate an empty list of pairs")
        
        rows = []
        for p in pairs:
            rows.append(self.encode_one(p.prompt, p.chosen))
        for p in pairs:
            rows.append(self.encode_one(p.prompt, p.rejected))
        
        B = len(pairs)
        L = max(len(r[0]) for r in rows)
        pad = self.tok.pad_token_id

        input_ids = torch.full((2 * B, L), pad, dtype = torch.long)
        attn = torch.zeros((2 * B, L), dtype = torch.long)
        comp = torch.zeros((2 * B, L), dtype = torch.long)

        for i, (ids, len_prompt, _, _) in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype = torch.long)
            attn[i, : len(ids)] = 1
            comp[i, len_prompt : len(ids)] = 1
        
        rep = self.report
        rep.n_pairs += B
        rep.n_prompt_truncated += sum(r[2] for r in rows)
        rep.n_response_truncated += sum(r[3] for r in rows)
        rep.max_len_seen = max(rep.max_len_seen, L)
        cl = comp.sum(dim = 1)
        rep.min_completion_len = (int(cl.min()) if rep.min_completion_len == 0 else min(rep.min_completion_len, int(cl.min())))

        if int(cl.min()) == 0:
            raise ConfigError("a row has zero completion tokens, log pi(y|x) would be 0.0, which the loss reads as probability 1. check max_length and the template")

        return DPOBatch(
            input_ids = input_ids, 
            attention_mask = attn, 
            completion_mask = comp, 
            B = B,
            uids = torch.tensor([pair_uid(p) for p in pairs], dtype = torch.int64),
            slices = [dict(p.slices) for p in pairs]
        )

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id


def render(rep : DPOCollateReport, width : int = 76) -> str:
    bar = "=" * width
    n = max(rep.n_pairs * 2, 1)
    L = [bar, "DPO COLLATE REPORT", bar,
        f"  tokenizer          : {rep.tokenizer}"
        + ("   (pad token added = eos)" if rep.pad_token_added else ""),
        f"  template           : {rep.template!r}",
        f"  max_length         : {rep.max_length}   padding: {rep.padding_side}",
        f"  append_eos         : {rep.append_eos}",
        f"  pairs seen         : {rep.n_pairs:,}   sequences: {rep.n_pairs * 2:,}",
        f"  longest seen       : {rep.max_len_seen}",
        f"  shortest completion: {rep.min_completion_len} tokens",
        f"  prompt truncated   : {rep.n_prompt_truncated:,} ({rep.n_prompt_truncated / n:.1%})",
        f"  response truncated : {rep.n_response_truncated:,} ({rep.n_response_truncated / n:.1%})",
        bar]

    if rep.n_response_truncated:
        L.insert(-1, "  !  RESPONSE TRUNCATED  the completion's tail was cut. for a dpo task that may be where the label lives. raise max_length.")
    return "\n".join(L)
