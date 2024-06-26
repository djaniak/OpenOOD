import logging
from typing import Any

import torch
import torch.nn as nn

from openood.postprocessors.base_postprocessor import BasePostprocessor


class PriorLogProbPostprocessor(BasePostprocessor):
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
        if net.stochastic_space == "H":
            kwargs = {"return_feature": True}
        elif net.stochastic_space == "Z":
            kwargs = {"return_embeddings": True}
        else:
            raise ValueError(
                "Cannot postprocess because there is no stochastic space in the model!"
            )
        logits, z3 = net(x3, **kwargs)
        _, pred = torch.max(logits, dim=1)

        if self.use_augumented:
            _, z1 = net(x1, **kwargs)
            _, z2 = net(x2, **kwargs)
            z1_logp = net.model.prior.log_prob(z1.unsqueeze(0)).sum(-1)
            z2_logp = net.model.prior.log_prob(z2.unsqueeze(0)).sum(-1)
            conf = z1_logp + z2_logp
        else:
            conf = net.model.prior.log_prob(z3.unsqueeze(0)).sum(-1)

        return pred, conf
