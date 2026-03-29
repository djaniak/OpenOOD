from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class FDBDPostprocessor(BasePostprocessor):
    """Feature-based DBD.

    Empirically, feature norm (||f||) can work better than distance to train mean
    (||f - mean_train||) as regularizer, so both are supported.
    """

    def __init__(self, config):
        super(FDBDPostprocessor, self).__init__(config)
        self.APS_mode = True
        self.setup_flag = False
        self.train_mean = None
        self.denominator_matrix = None
        self.num_classes = None
        self.activation_log = None

        self.args = self.config.postprocessor.postprocessor_args
        self.distance_as_normalizer = self.args.distance_as_normalizer
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
            "fDBD requires classifier weights, but could not find fc/get_fc/get_fc_layer."
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

            weight = self._get_fc_weight(net, feature_device)
            self.num_classes = weight.shape[0]

            denominator_matrix = np.zeros((self.num_classes, self.num_classes))
            weight_np = weight.detach().cpu().numpy()
            for p in range(self.num_classes):
                weight_delta = weight_np - weight_np[p, :]
                denominator = np.linalg.norm(weight_delta, axis=1)
                denominator[p] = 1.0
                denominator_matrix[p, :] = denominator

            self.denominator_matrix = torch.as_tensor(
                denominator_matrix, dtype=torch.float32, device=feature_device
            )
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, preembedded: bool):
        output, feature = net(data, return_feature=True, preembedded=preembedded)
        values, pred = output.max(1)
        logits_sub = torch.abs(output - values.unsqueeze(1))
        denominator = self.denominator_matrix[pred].to(
            device=feature.device, dtype=feature.dtype
        )
        numerator = torch.sum(logits_sub / denominator, dim=1)

        if self.distance_as_normalizer:
            normalizer = torch.norm(
                feature - self.train_mean.to(device=feature.device, dtype=feature.dtype),
                dim=1,
            )
        else:
            normalizer = torch.norm(feature, dim=1)

        score = numerator / normalizer.clamp_min(1e-12)
        return pred, score

    def set_hyperparam(self, hyperparam: list):
        self.distance_as_normalizer = hyperparam[0]

    def get_hyperparam(self):
        return self.distance_as_normalizer


# Backward-compatible alias with common naming in external snippets.
fDBDPostprocessor = FDBDPostprocessor
