from dataclasses import dataclass
from typing import Literal
import torch
from rlhf.core.contracts import ConfigError
from rlhf.core.losses import bt_loss


@dataclass
class DPOLossMetrics:
    loss : float = 0.0
    n_pairs : int  = 0

    delta_mean : float = 0.0
    accuracy : float = 0.0
    reward_chosen : float = 0.0
    reward_rejected : float = 0.0
    reward_margin : float = 0.0

    grad_weight_mean : float = 0.0
    frac_pairs_saturated : float = 0.0

    policy_logp_chosen : float = 0.0
    policy_logp_rejected : float = 0.0
    ref_logp_chosen : float = 0.0
    ref_logp_rejected : float = 0.0
    logratio_chosen : float = 0.0
    logratio_rejected : float = 0.0

    sft_term : float = 0.0

    def to_dict(self):
        return dict(self.__dict__)


def dpo_loss(
    policy_chosen_logps : torch.Tensor,
    policy_rejected_logps : torch.Tensor,
    ref_chosen_logps : torch.Tensor,
    ref_rejected_logps : torch.Tensor,
    beta : float = 0.1,
    sft_wt : float = 0.0,
    reduction : Literal["mean", "sum", "none"] = "mean"
) -> tuple:

    if beta <= 0.0:
        raise ConfigError(f"beta must be > 0 but instead is {beta}")
    
    if sft_wt < 0.0:
        raise ConfigError(f"sft_wt must be >= 0 but instead is {sft_wt}")
    
    shapes = [t.shape for t in (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)]
    if any(len(s) != 1 for s in shapes) or len(set(shapes)) != 1:
        raise ConfigError(f"all four inputs must be [B] and the same length, got {shapes}")

    pc = policy_chosen_logps.float()
    pr = policy_rejected_logps.float()
    rc = ref_chosen_logps.float()
    rr= ref_rejected_logps.float()

    rc, rr = rc.detach(), rr.detach()
    logratio_chosen = pc - rc
    logratio_rejected = pr - rr
    reward_chosen = beta * logratio_chosen
    reward_rejected = beta * logratio_rejected
    loss = bt_loss(reward_chosen, reward_rejected, reduction = reduction)

    delta = (reward_chosen - reward_rejected).detach()
    grad_weight = torch.sigmoid(-delta)

    sft = torch.zeros((), device = pc.device)
    if sft_wt > 0:
        sft = - sft_wt * pc.mean()
        loss = loss + sft
    
    dpo_metrics = DPOLossMetrics(
        loss = float(loss.detach()),
        n_pairs = len(pc),
        delta_mean = float(delta.mean()),
        accuracy = float((delta > 0).float().mean()),
        reward_chosen = float(reward_chosen.detach().mean()),
        reward_rejected = float(reward_rejected.detach().mean()),
        reward_margin = float((reward_chosen - reward_rejected).detach().mean()),
        grad_weight_mean = float(grad_weight.mean()),
        frac_pairs_saturated = float((grad_weight < 0.01).float().mean()),
        policy_logp_chosen = float(pc.detach().mean()),
        policy_logp_rejected = float(pr.detach().mean()),
        ref_logp_chosen = float(rc.mean()),
        ref_logp_rejected = float(rr.mean()),
        logratio_chosen = float(logratio_chosen.detach().mean()),
        logratio_rejected = float(logratio_rejected.detach().mean()),
        sft_term = float(sft.detach())
    )

    return loss, dpo_metrics


def render(m : DPOLossMetrics, width : int = 76) -> str:
    bar = "=" * width
    return "\n".join([
        bar, "DPO LOSS", bar,
        f"  loss              : {m.loss:8.4f}   over {m.n_pairs} pairs",
        f"  Delta (mean)      : {m.delta_mean:+8.4f}   accuracy {m.accuracy:.3f}",
        f"  implicit reward   : chosen {m.reward_chosen:+7.4f}   rejected {m.reward_rejected:+7.4f}"
        f"  margin {m.reward_margin:+7.4f}",
        f"  grad weight       : {m.grad_weight_mean:8.4f}   saturated {m.frac_pairs_saturated:.1%}",
        f"  policy log pi     : chosen {m.policy_logp_chosen:9.3f}   rejected {m.policy_logp_rejected:9.3f}",
        f"  log ratio to ref  : chosen {m.logratio_chosen:+9.3f}   rejected {m.logratio_rejected:+9.3f}",
        bar,
    ])
