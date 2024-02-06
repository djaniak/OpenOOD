from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class SigmaStdPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, dist = net(data, return_dist=True)
        _, pred = torch.max(logits, dim=1)
        (_, log_var) = dist
        sigma = torch.exp(0.5 * log_var)
        conf = -sigma.std(dim=1)
        return pred, conf
