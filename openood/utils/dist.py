from typing import Literal, Optional

import numpy as np
import torch


def log_normal_diag(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: Optional[Literal["avg", "mean"]] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    log_prob = -0.5 * (
        torch.log(2 * torch.tensor(np.pi)) + log_var + torch.exp(-log_var) * (x - mu) ** 2
    )
    if reduction == "avg":
        return torch.mean(log_prob, dim)
    elif reduction == "sum":
        return torch.sum(log_prob, dim)
    else:
        return log_prob