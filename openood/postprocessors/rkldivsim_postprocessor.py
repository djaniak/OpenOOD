from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl_divergence
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class RKlDivSimPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False

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
                        x1, x2, x3 = batch["data"]
                        data = (x1.cuda(), x2.cuda(), x3.cuda())
                    else:
                        data = batch["data"].cuda()
                    labels = batch["label"]
                    logits, (mus, logvars) = net(data, return_dist=True)
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

            self.train_q = Independent(Normal(all_mus, all_logvars), 1)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, (mus, logvars) = net(data, return_dist=True)
        pred = logits.argmax(1)

        conf = []
        for mu, logvar in tqdm(zip(mus, logvars)):
            q = Independent(Normal(mu, logvar), 1)
            min_kldiv = torch.min(kl_divergence(q, self.train_q))
            conf.append(min_kldiv)
        conf = torch.hstack(conf).cpu()

        return pred, conf
