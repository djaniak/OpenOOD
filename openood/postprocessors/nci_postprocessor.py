from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NCIPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NCIPostprocessor, self).__init__(config)
        self.APS_mode = True
        self.setup_flag = False
        self.train_mean = None
        self.w = None
        self.activation_log = None

        self.args = self.config.postprocessor.postprocessor_args
        self.alpha = self.args.alpha
        self.args_dict = getattr(
            self.config.postprocessor, "postprocessor_sweep", {}
        )

    def _get_fc_weight(self, net: nn.Module, device: torch.device) -> torch.Tensor:
        module = net.module if hasattr(net, "module") else net

        if hasattr(module, "get_fc_layer"):
            fc = module.get_fc_layer()
            if hasattr(fc, "weight"):
                return fc.weight.detach().to(device=device, dtype=torch.float32)

        if hasattr(module, "fc") and hasattr(module.fc, "weight"):
            return module.fc.weight.detach().to(device=device, dtype=torch.float32)

        if hasattr(module, "get_fc"):
            weight, _ = module.get_fc()
            return torch.as_tensor(weight, dtype=torch.float32, device=device)

        raise AttributeError(
            "NCI requires classifier weights, but could not find fc/get_fc/get_fc_layer."
        )

    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
        preembedded=False,
        *args,
        **kwargs,
    ):
        if not self.setup_flag:
            activation_log = []
            feature_device = None
            net.eval()
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["train"], desc="Setup: ", position=0, leave=True
                ):
                    data = batch["data"].cuda()
                    data = data.float()

                    _, feature = net(
                        data, return_feature=True, preembedded=preembedded
                    )
                    feature_device = feature.device
                    activation_log.append(feature.detach().cpu().numpy())

            activation_log_concat = np.concatenate(activation_log, axis=0)
            self.activation_log = activation_log_concat
            self.train_mean = torch.from_numpy(
                np.mean(activation_log_concat, axis=0)
            ).to(device=feature_device, dtype=torch.float32)

            self.w = self._get_fc_weight(net, feature_device)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, preembedded: bool):
        output, feature = net(data, return_feature=True, preembedded=preembedded)
        _, pred = output.max(1)

        train_mean = self.train_mean.to(device=feature.device, dtype=feature.dtype)
        weight = self.w.to(device=feature.device, dtype=feature.dtype)

        centered_feature = feature - train_mean
        aligned_projection = torch.sum(weight[pred] * centered_feature, dim=1)
        centered_norm = torch.norm(centered_feature, dim=1).clamp_min(1e-12)
        score = aligned_projection / centered_norm + self.alpha * torch.norm(
            feature, p=1, dim=1
        )
        return pred, score

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]

    def get_hyperparam(self):
        return self.alpha
