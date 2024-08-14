import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class MultiMDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        print(config)
        self.setup_flag = False
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.n_last_layers = self.config.postprocessor.n_last_layers
        self.class_mean = {}
        self.precision = {}

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict, preembedded=False, *args, **kwargs):
        if self.setup_flag:
            return
        print("\n Estimating mean and variance from training set...")
        all_feats, all_labels, all_preds = self._collect_features_and_labels(net, id_loader_dict, preembedded)
        _, num_layers, _= all_feats.shape
        self.layers = list(range(num_layers-self.n_last_layers, num_layers))
        for layer_idx in tqdm(self.layers, desc="Computing class statistics for each layer"):
            self._compute_class_statistics(all_feats, all_labels, layer_idx)
        self.setup_flag = True

    def _collect_features_and_labels(self, net, id_loader_dict, preembedded):
        all_feats, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict["train"], desc="Collecting features and labels"):
                data, labels = batch["data"].cuda(), batch["label"]
                logits, features = net(data, return_feature=True, preembedded=preembedded)
                all_feats.append(features.cpu())
                all_labels.append(labels)
                all_preds.append(logits.argmax(1).cpu())
        return torch.cat(all_feats), torch.cat(all_labels), torch.cat(all_preds)

    def _compute_class_statistics(self, feats, labels, layer_idx):
        feats = feats[:,layer_idx,:]
        _class_mean, centered_data = [], []
        for c in range(self.num_classes):
            class_samples = feats[labels.eq(c)].data
            _class_mean.append(class_samples.mean(0))
            centered_data.append(class_samples - _class_mean[c].view(1, -1))
        self.class_mean[layer_idx] = torch.stack(_class_mean)
        group_lasso = EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
        self.precision[layer_idx] =torch.from_numpy(group_lasso.precision_).float()

    @torch.no_grad()
    def postprocess(self, net, data, preembedded):
        logits, all_features = net(data, return_feature=True, preembedded=preembedded)
        pred = logits.argmax(1)
        layer_scores = self._compute_layer_scores(logits, all_features)
        return pred, layer_scores.sum(dim=1)

    def _compute_layer_scores(self, logits, all_features):
        batch_size = all_features.shape[0]
        layer_scores = torch.empty(batch_size, len(self.layers))
        for i, layer_idx in enumerate(self.layers):
            features = all_features[:,layer_idx,:]
            class_scores = torch.zeros((logits.shape[0], self.num_classes))
            for c in range(self.num_classes):
                # tensor = features.cpu() - self.class_mean[layer_idx][c].view(1, -1)
                tensor = features - self.class_mean[layer_idx][c].view(1, -1).cuda()
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision[layer_idx].cuda()), tensor.t()
                ).diag().cpu()
            conf = torch.max(class_scores, dim=1)[0]
            layer_scores[:, i] = conf
        return layer_scores
