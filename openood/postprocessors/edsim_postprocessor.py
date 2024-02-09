from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from tqdm import tqdm

from openood.postprocessors.base_postprocessor import BasePostprocessor


class EuclideanDistSimPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.setup_flag = False
        self.APS_mode = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            all_features = []
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
                    logits, feats = net(x.cuda(), return_feature=True)
                    all_features.append(feats.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_features = torch.cat(all_features)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f" Train acc: {train_acc:.2%}")

            self.train_feats = all_features
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, feats = net(data, return_feature=True)
        pred = logits.argmax(1)

        dist = torch.tensor(cdist(feats, self.train_feats))
        conf = torch.min(dist, dim=1).values

        return pred, conf
