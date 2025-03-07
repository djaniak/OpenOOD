from typing import Callable, List, Optional, Any

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet

from .datasets import DATASET_METADATA, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = './data',
        config_root: str = './configs',
        preprocessor: Optional[Any] = None,
        postprocessor_name: Optional[str] = None,
        postprocessor: Optional[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        id_preembedded_dir: Optional[str] = None,
        selected_ood_datasets: Optional[List[str]] = None,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module): The base classifier.
            id_name (str): The name of the in-distribution dataset.
            data_root (str, optional): The path of the data folder. Defaults to './data'.
            config_root (str, optional): The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional): The preprocessor of input images. Defaults to None.
            postprocessor_name (str, optional): The name of the postprocessor that obtains OOD score. Defaults to None.
            postprocessor (Type[BasePostprocessor], optional): An actual postprocessor instance. Defaults to None.
            batch_size (int, optional): The batch size of samples. Defaults to 200.
            shuffle (bool, optional): Whether shuffling samples. Defaults to False.
            id_preembedded_dir (str, optional): The directory of pre-embedded ID dataset. Defaults to None.
            selected_ood_datasets (list[str], optional): The list of OOD datasets to evaluate. Defaults to None.
            num_workers (int, optional): The num_workers argument that will be passed to data loaders. Defaults to 4.

        Raises:
            ValueError: If both postprocessor_name and postprocessor are None.
            ValueError: If the specified ID dataset {id_name} is not supported.
            TypeError: If the passed postprocessor does not inherit BasePostprocessor.
        """
        self.id_preembedded_dir = id_preembedded_dir
        self.preembedded = id_preembedded_dir is not None

        self._validate_args(postprocessor_name, postprocessor, id_name)
        self.preprocessor = preprocessor or get_default_preprocessor(id_name)
        self.config_root = config_root or self._get_default_config_root()
        self.postprocessor = postprocessor or get_postprocessor(self.config_root, postprocessor_name, id_name)
        self._validate_postprocessor()

        data_setup(data_root, id_name)
        self.dataloader_dict = self._get_dataloader_dict(id_name, data_root, batch_size, shuffle, num_workers, selected_ood_datasets)
        self.net = self._wrap_net_with_postprocessor(net, postprocessor_name)
        self.postprocessor.setup(self.net, self.dataloader_dict['id'], self.dataloader_dict['ood'], preembedded=self.preembedded)

        self.id_name = id_name
        self.metrics = self._initialize_metrics()
        self.scores = self._initialize_scores()

        if self.postprocessor.APS_mode and not self.postprocessor.hyperparam_search_done:
            self.hyperparam_search()

        self.net.eval()

    def _validate_args(self, postprocessor_name, postprocessor, id_name):
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print('Postprocessor_name is ignored because postprocessor is passed')
        if id_name not in DATASET_METADATA:
            raise ValueError(f'Dataset [{id_name}] is not supported')

    def _get_default_config_root(self):
        filepath = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(*filepath.split('/')[:-2], 'configs')

    def _validate_postprocessor(self):
        if not isinstance(self.postprocessor, BasePostprocessor):
            raise TypeError('postprocessor should inherit BasePostprocessor in OpenOOD')

    def _get_dataloader_dict(self, id_name, data_root, batch_size, shuffle, num_workers, selected_ood_datasets):
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        return get_id_ood_dataloader(
            id_name, 
            data_root, 
            self.preprocessor, 
            id_preembedded_dir=self.id_preembedded_dir,
            selected_ood_datasets=selected_ood_datasets, 
            **loader_kwargs
        )

    def _wrap_net_with_postprocessor(self, net, postprocessor_name):
        if postprocessor_name == 'react':
            return ReactNet(net)
        elif postprocessor_name == 'ash':
            return ASHNet(net)
        elif postprocessor_name == 'scale':
            return ScaleNet(net)
        return net

    def _initialize_metrics(self):
        return {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }

    def _initialize_scores(self):
        return {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None for k in self.dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near': {k: None for k in self.dataloader_dict['ood']['near'].keys()},
                'far': {k: None for k in self.dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None for k in self.dataloader_dict['csid'].keys()},
            'csid_labels': {k: None for k in self.dataloader_dict['csid'].keys()},
        }

    def _classifier_inference(self, data_loader: DataLoader, msg: str = 'Acc Eval', progress: bool = True):
        self.net.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch["data"] if not isinstance(batch['data'], list) else batch["data"][-1]
                logits = self.net(data.cuda(), return_feature=False, preembedded=self.preembedded)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(batch['label'])

        return torch.cat(all_preds), torch.cat(all_labels)

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name not in ['id', 'csid']:
            raise ValueError(f'Unknown data name {data_name}')

        if self.metrics[f'{data_name}_acc'] is not None:
            return self.metrics[f'{data_name}_acc']

        if data_name == 'id':
            return self._evaluate_id_acc()
        else:
            return self._evaluate_csid_acc()

    def _evaluate_id_acc(self):
        if self.scores['id_preds'] is None:
            all_preds, all_labels = self._classifier_inference(self.dataloader_dict['id']['test'], 'ID Acc Eval')
            self.scores['id_preds'], self.scores['id_labels'] = all_preds, all_labels
        else:
            all_preds, all_labels = self.scores['id_preds'], self.scores['id_labels']

        correct = (all_preds == all_labels).sum().item()
        acc = correct / len(all_labels) * 100
        self.metrics['id_acc'] = acc
        return acc

    def _evaluate_csid_acc(self):
        correct, total = 0, 0
        for dataname, dataloader in self.dataloader_dict['csid'].items():
            if self.scores['csid_preds'][dataname] is None:
                all_preds, all_labels = self._classifier_inference(dataloader, f'CSID {dataname} Acc Eval')
                self.scores['csid_preds'][dataname], self.scores['csid_labels'][dataname] = all_preds, all_labels
            else:
                all_preds, all_labels = self.scores['csid_preds'][dataname], self.scores['csid_labels'][dataname]

            correct += (all_preds == all_labels).sum().item()
            total += len(all_labels)

        if self.scores['id_preds'] is None:
            all_preds, all_labels = self._classifier_inference(self.dataloader_dict['id']['test'], 'ID Acc Eval')
            self.scores['id_preds'], self.scores['id_labels'] = all_preds, all_labels
        else:
            all_preds, all_labels = self.scores['id_preds'], self.scores['id_labels']

        correct += (all_preds == all_labels).sum().item()
        total += len(all_labels)

        acc = correct / total * 100
        self.metrics['csid_acc'] = acc
        return acc

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        task = 'fsood' if fsood else 'ood'
        id_name = 'csid' if fsood else 'id'

        if self.metrics[task] is not None:
            print('Evaluation has already been done!')
            return self.metrics[task]

        self.net.eval()
        id_pred, id_conf, id_gt = self._get_id_scores(progress)

        if fsood:
            id_pred, id_conf, id_gt = self._get_combined_id_scores(id_pred, id_conf, id_gt, progress)

        near_metrics = self._process_ood_split(id_pred, id_conf, id_gt, 'near', progress)
        far_metrics = self._process_ood_split(id_pred, id_conf, id_gt, 'far', progress)

        if self.metrics[f'{id_name}_acc'] is None:
            self.eval_acc(id_name)

        combined_metrics = self._combine_metrics(near_metrics, far_metrics, id_name)
        self.metrics[task] = pd.DataFrame(
            combined_metrics,
            index=self._get_indices(near_metrics, far_metrics),
            columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
        )

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            print(self.metrics[task])

        return self.metrics[task]

    def _get_id_scores(self, progress):
        if self.scores['id']['test'] is None:
            print(f'Performing inference on {self.id_name} test set...', flush=True)
            id_pred, id_conf, id_gt = self.postprocessor.inference(self.net, self.dataloader_dict['id']['test'], preembedded=self.preembedded, progress=progress)
            self.scores['id']['test'] = [id_pred, id_conf, id_gt]
        else:
            id_pred, id_conf, id_gt = self.scores['id']['test']
        return id_pred, id_conf, id_gt

    def _get_combined_id_scores(self, id_pred, id_conf, id_gt, progress):
        csid_pred, csid_conf, csid_gt = [], [], []
        for i, dataset_name in enumerate(self.scores['csid'].keys()):
            if self.scores['csid'][dataset_name] is None:
                print(f'Performing inference on {self.id_name} (cs) test set [{i+1}]: {dataset_name}...', flush=True)
                temp_pred, temp_conf, temp_gt = self.postprocessor.inference(self.net, self.dataloader_dict['csid'][dataset_name], preembedded=False, progress=progress)
                self.scores['csid'][dataset_name] = [temp_pred, temp_conf, temp_gt]

            csid_pred.append(self.scores['csid'][dataset_name][0])
            csid_conf.append(self.scores['csid'][dataset_name][1])
            csid_gt.append(self.scores['csid'][dataset_name][2])

        csid_pred = np.concatenate(csid_pred)
        csid_conf = np.concatenate(csid_conf)
        csid_gt = np.concatenate(csid_gt)

        id_pred = np.concatenate((id_pred, csid_pred))
        id_conf = np.concatenate((id_conf, csid_conf))
        id_gt = np.concatenate((id_gt, csid_gt))

        return id_pred, id_conf, id_gt

    def _process_ood_split(self, id_pred, id_conf, id_gt, ood_split, progress):
        if len(self.dataloader_dict['ood'][ood_split]) == 0:
            return None

        print(f'Processing {ood_split} ood...', flush=True)
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...', flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(self.net, ood_dl, preembedded=False, progress=progress)
                self.scores['ood'][ood_split][dataset_name] = [ood_pred, ood_conf, ood_gt]
            else:
                print(f'Inference has been performed on {dataset_name} dataset...', flush=True)
                ood_pred, ood_conf, ood_gt = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        if len(metrics_list) > 1:
            print('Computing mean metrics...', flush=True)
            metrics_list = np.array(metrics_list)
            metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
            self._print_metrics(list(metrics_mean[0]))
            return np.concatenate([metrics_list, metrics_mean], axis=0) * 100
        else:
            return np.array(metrics_list) * 100

    def _combine_metrics(self, near_metrics, far_metrics, id_name):
        all_metrics = []
        if near_metrics is not None:
            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] * len(near_metrics))
            all_metrics.append(near_metrics)
        if far_metrics is not None:
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] * len(far_metrics))
            all_metrics.append(far_metrics)
        return np.concatenate(all_metrics, axis=0) if len(all_metrics) > 1 else all_metrics[0]

    def _get_indices(self, near_metrics, far_metrics):
        indices = []
        if near_metrics is not None:
            indices.extend(list(self.dataloader_dict['ood']['near'].keys()))
            if len(self.dataloader_dict['ood']['near']) > 1:
                indices.append('nearood')
        if far_metrics is not None:
            indices.extend(list(self.dataloader_dict['ood']['far'].keys()))
            if len(self.dataloader_dict['ood']['far']) > 1:
                indices.append('farood')
        return indices

    def _print_metrics(self, metrics):
        fpr, auroc, aupr_in, aupr_out, _ = metrics
        print(f'FPR@95: {100 * fpr:.2f}, AUROC: {100 * auroc:.2f}', end=' ', flush=True)
        print(f'AUPR_IN: {100 * aupr_in:.2f}, AUPR_OUT: {100 * aupr_out:.2f}', flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = list(self.postprocessor.args_dict.keys())
        hyperparam_list = [self.postprocessor.args_dict[name] for name in hyperparam_names]
        hyperparam_combination = self.recursive_generator(hyperparam_list, len(hyperparam_names))

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = self.postprocessor.inference(self.net, self.dataloader_dict['id']['val'], preembedded=self.preembedded)
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(self.net, self.dataloader_dict['ood']['val'], preembedded=False)

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print(f'Hyperparam: {hyperparam}, auroc: {auroc}')
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print(f'Final hyperparam: {self.postprocessor.get_hyperparam()}')
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        if n == 1:
            return [[x] for x in list[0]]
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    results.append(y + [x])
            return results
