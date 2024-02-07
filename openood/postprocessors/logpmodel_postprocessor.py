import logging
from typing import Any

import torch
import torch.nn as nn

from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.utils.dist import log_normal_diag


class ModelLogProbPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        logging.warning(
            "Remeber: use this postprocessor with BarlowTwins preprocessor!"
        )
        super().__init__(None)
        self.use_augumented = config.postprocessor.use_augumented
        self.APS_mode = False

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        x1, x2, x3 = data
        kwargs = {"return_dist": True}
        if net.stochastic_space == "H":
            kwargs["return_feature"] = True
        elif net.stochastic_space == "Z":
            kwargs["return_embeddings"] = True
        else:
            raise ValueError(
                "Cannot postprocess because there is no stochastic space in the model!"
            )

        logits, z3, (mu3, logvar3) = net(x3, **kwargs)
        _, pred = torch.max(logits, dim=1)

        if self.use_augumented:
            _, z1, (mu1, logvar1) = net(x1, **kwargs)
            _, z2, (mu2, logvar2) = net(x2, **kwargs)
            conf = self.log_p(z1, mu1, logvar1) + self.log_p(z2, mu2, logvar2)
        else:
            conf = self.log_p(z3, mu3, logvar3)

        return pred, conf

    def log_p(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return log_normal_diag(x.unsqueeze(0), mu, logvar, reduction="avg", dim=0).sum(
            -1
        )
