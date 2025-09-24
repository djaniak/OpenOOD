from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


def normalizer(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.activation_log_torch = None
        self.index = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict: dict, preembedded: bool = False):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in id_loader_dict['train']:
                    data = batch['data'].cuda()
                    _, feature = net(data, return_feature=True, preembedded=preembedded)
                    activation_log.append(normalizer(feature.data.cpu().numpy()))
            self.activation_log = np.concatenate(activation_log, axis=0).astype(np.float32)
            self.index = faiss.IndexFlatL2(self.activation_log.shape[1])
            self.index.add(self.activation_log)
            self.activation_log_torch = torch.from_numpy(self.activation_log).float().cuda()
            self.setup_flag = True

    # @torch.no_grad()
    # def postprocess_faiss(self, net, data, preembedded=False):
    #     output, feature = net(data, return_feature=True, preembedded=preembedded)
    #     feature_normed = normalizer(feature.data.cpu().numpy().astype(np.float32))
    #     D, _ = self.index.search(feature_normed, self.K)
    #     kth_dist = -D[:, -1]
    #     _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
    #     return pred, torch.from_numpy(kth_dist)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, preembedded: bool = False):
        output, feature = net(data, return_feature=True, preembedded=preembedded)
        feature_normed_np = normalizer(feature.data.cpu().numpy().astype(np.float32))
        feature_normed = torch.from_numpy(feature_normed_np).float().cuda()
        # Use squared L2 distance to match FAISS
        dists = torch.cdist(feature_normed, self.activation_log_torch, p=2) ** 2
        kth_dist, _ = torch.topk(dists, self.K, largest=False)
        kth_dist = -kth_dist[:, -1].cpu()
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, kth_dist

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
