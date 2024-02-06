from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader, progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(
            data_loader, disable=not progress or not comm.is_main_process()
        ):
            if isinstance(batch["data"], list):
                x1, x2, x3 = batch["data"]
                data = (x1.cuda(), x2.cuda(), x3.cuda())
            else:
                data = batch["data"].cuda()
            label = batch["label"].cuda()
            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
