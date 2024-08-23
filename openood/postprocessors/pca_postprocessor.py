from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cuml import PCA as cumlPCA
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class PCAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.n_component = self.postprocessor_args.n_component
        self.pca = cumlPCA(n_components=self.n_component)

        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, preembedded=False, *args, **kwargs):
        if not self.setup_flag:
            print("\n Estimating mean and variance from training set...")
            all_feats = []
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["train"], desc="Setup: ", position=0, leave=True
                ):
                    data, labels = batch["data"].cuda(), batch["label"]
                    logits, features = net(data, return_feature=True, preembedded=preembedded)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f" Train acc: {train_acc:.2%}")

            # fit PCA
            self.pca.fit(all_feats.numpy())
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, preembedded: bool):
        logits, features = net(data, return_feature=True, preembedded=preembedded)
        features = features.cpu().numpy()
        pred = logits.argmax(1)

        projected = self.pca.transform(features)
        # reconstruct the features from the projection
        reconstructed = self.pca.inverse_transform(projected)
        # calculate the reconstruction error (L2 norm)
        conf = np.linalg.norm(features - reconstructed, axis=1)
        conf = torch.tensor(conf)

        return pred, conf
