from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from openood.postprocessors.base_postprocessor import BasePostprocessor


class BhattacharyyaDistSimPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.APS_mode = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            all_mus = []
            all_logvars = []
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for batch in tqdm(
                        id_loader_dict["train"], desc="Setup: ", position=0, leave=True
                ):
                    if isinstance(batch["data"], list):
                        _, _, x = batch["data"]
                    else:
                        x = batch["data"]
                    labels = batch["label"]
                    logits, (mus, logvars) = net(x.cuda(), return_dist=True)
                    all_mus.append(mus)
                    all_logvars.append(logvars)
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_mus = torch.cat(all_mus)
            all_logvars = torch.cat(all_logvars)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f" Train acc: {train_acc:.2%}")

            self.qm = all_mus
            self.qv = torch.exp(0.5 * all_logvars)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        x1, x2, x3 = data
        logits, (mus, logvars) = net(x3, return_dist=True)
        pred = logits.argmax(1)

        conf = []
        for pm, pv in zip(mus, logvars):
            pv = torch.exp(0.5 * pv)

            bh_dists = self.gau_bh(pm, pv, self.qm, self.qv)
            bh_dists = torch.nan_to_num(bh_dists, nan=torch.finfo(torch.float16).max)
            min_bh_dist = torch.min(bh_dists)
            conf.append(min_bh_dist)
        conf = torch.hstack(conf).cpu()

        return pred, conf

    @staticmethod
    def gau_bh(pm, pv, qm, qv):
        """
        Classification-based Bhattacharyya distance between two Gaussians with diagonal covariance.

        From https://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
        """
        if (len(qm.shape) == 2):
            dim = 1
        else:
            dim = 0
        # Difference between means pm, qm
        diff = qm - pm
        # Interpolated variances
        pqv = (pv + qv) / 2.
        # Log-determinants of pv, qv
        ldpv = torch.log(pv).sum()
        ldqv = torch.log(qv).sum(dim)
        # Log-determinant of pqv
        ldpqv = torch.log(pqv).sum(dim)
        # "Shape" component (based on covariances only)
        # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
        norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
        # "Divergence" component (actually just scaled Mahalanobis distance)
        # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
        dist = 0.125 * (diff * (1./pqv) * diff).sum(dim)
        return dist + norm
