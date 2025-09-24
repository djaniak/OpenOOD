from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, preembedded=False, *args, **kwargs):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b = net.get_fc()
                # Convert to tensor if needed
                if isinstance(self.w, np.ndarray):
                    self.w = torch.from_numpy(self.w)
                if isinstance(self.b, np.ndarray):
                    self.b = torch.from_numpy(self.b)
                self.w = self.w.cuda()
                self.b = self.b.cuda()
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ',
                                position=0,
                                leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net(data, return_feature=True, preembedded=preembedded)
                    feature_id_train.append(feature)
                feature_id_train = torch.cat(feature_id_train, dim=0)  # stays on GPU
                logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -torch.linalg.pinv(self.w) @ self.b
            centered = feature_id_train - self.u
            cov = torch.cov(centered.T)
            eig_vals, eigen_vectors = torch.linalg.eigh(cov)
            idx = torch.argsort(eig_vals * -1)[self.dim:]
            self.NS = eigen_vectors[:, idx].contiguous()

            vlogit_id_train = torch.norm((centered @ self.NS), dim=-1)
            self.alpha = logit_id_train.max(dim=-1).values.mean() / vlogit_id_train.mean()
            print(f'{self.alpha:.4f}')

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, preembedded: bool):
        _, feature_ood = net(data, return_feature=True, preembedded=preembedded)
        # Ensure everything is on the same device as self.w
        device = self.w.device
        feature_ood = feature_ood.to(device)
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        # Use PyTorch for logsumexp and norm
        energy_ood = torch.logsumexp(logit_ood, dim=-1)
        vlogit_ood = torch.norm((feature_ood - self.u) @ self.NS, dim=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return pred, score_ood

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
