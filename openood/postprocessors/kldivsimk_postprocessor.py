from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl_divergence
from tqdm import tqdm

from openood.postprocessors.base_postprocessor import BasePostprocessor


class KlDivSimKPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KlDivSimKPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.args_dict = self.config.postprocessor.postprocessor_sweep
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

            self.train_q = Independent(Normal(all_mus, torch.exp(0.5 * all_logvars)), 1)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        x1, x2, x3 = data
        logits, (mus, logvars) = net(x3, return_dist=True)
        pred = logits.argmax(1)

        conf = []
        for mu, logvar in zip(mus, logvars):
            q = Independent(Normal(mu, torch.exp(0.5 * logvar)), 1)
            kl_divs = kl_divergence(self.train_q, q)
            kl_divs = torch.nan_to_num(kl_divs, nan=torch.finfo(torch.float16).max)
            k_min_kldiv = torch.sort(kl_divs)[0][self.K]
            conf.append(k_min_kldiv)
        conf = torch.hstack(conf).cpu()

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
