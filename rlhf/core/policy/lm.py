from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Literal
import torch
import torch.nn.functional as F
from torch import nn
from rlhf.core.config import ConfigError


@dataclass
class PolicyReport:
    source : str
    num_params : int
    device : str = "cpu"
    dtype : str = "float32"
    vocab_size : int = 0

    def to_dict(self):
        return dict(self.__dict__)


def load_policy(
    model_name : Optional[str] = None,
    model_ckpt : Optional[str] = None,
    device : str = "cpu",
    dtype : str = "float32",
    vocab_size : Optional[int] = None
):
    if model_name is not None and model_ckpt is not None:
        raise ConfigError(f"Can only pass one among model_name : {model_name} or model_ckpt : {model_ckpt}")
    if model_name is None and model_ckpt is None:
        raise ConfigError(f"no model info provided, pass either model_name or model_ckpt")
    
    from transformers import AutoModelForCausalLM
    src = model_name or model_ckpt
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(src, dtype = torch_dtype).to(device)
    if vocab_size is None:
        try:
            vocab_size = model.config.vocab_size
        except:
            raise ConfigError("vocab_size can not be determined, pls pass vocab_size for the model separately.")
    
    rep = PolicyReport(
        source = src,
        num_params = sum(p.numel() for p in model.parameters()),
        device = str(device), 
        dtype = dtype,
        vocab_size = vocab_size
    )
    return model, rep


def freeze(model : nn.Module):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def disable_dropout(model : nn.Module):
    cnt = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            cnt += 1
    return cnt


def sequence_logprobs(
    model : nn.Module,
    input_ids : torch.Tensor,
    attention_mask : torch.Tensor,
    completion_mask : torch.Tensor,
    length_norm : Literal["sum", "mean"] = "sum",
    logits : Optional[torch.Tensor] = None
):
    if length_norm not in ("sum", "mean"):
        raise ConfigError(f"length_norm must be one among the 'sum', 'mean'")
    
    if logits is None:
        logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
    
    shift_labels = input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_comp_mask = completion_mask[:, 1:]
    picked_logits = shift_logits.gather(dim = -1, index = shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = picked_logits - torch.logsumexp(shift_logits, dim = -1)
    token_logp = torch.where(shift_comp_mask.bool(), token_logp, torch.zeros_like(token_logp))
    total = token_logp.sum(dim = -1)
    if length_norm == "mean":
        total = total / shift_comp_mask.sum(dim = -1).clamp(min = 1)
    return total


@torch.inference_mode()
def generate(
    model : nn.Module,
    tokenizer,
    prompts : Sequence[str],
    num_samples_per_prompt : int = 1,
    temperature : float = 1.0,
    top_p : float = 0.95,
    max_new_tokens : int = 48,
    seed : Optional[int] = None
):
    if not prompts:
        raise ConfigError("generate needs atleast one prompt")
    
    was = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        if seed is not None:
            torch.manual_seed(seed)
        enc = tokenizer(list(prompts), return_tensors = "pt", padding = True, add_special_tokens = False).to(model.device)
        out = model.generate(
                **enc,
                do_sample = True, 
                temperature = temperature, 
                top_p = top_p,
                max_new_tokens = max_new_tokens,
                num_return_sequences = num_samples_per_prompt,
                pad_token_id = tokenizer.pad_token_id
            )
        
        gen = out[:, enc["input_ids"].shape[1]: ]
        text = tokenizer.batch_decode(gen, skip_special_tokens = True)
        k = num_samples_per_prompt
        return [text[i * k : (i + 1) * k] for i in range(len(prompts))]
    finally:
        tokenizer.padding_side = was
