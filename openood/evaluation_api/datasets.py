import logging
import os
from pathlib import Path

import gdown
import zipfile

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from sklearn.model_selection import train_test_split

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor

from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_METADATA = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

DOWNLOAD_ID_DICT = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP'
}

DIR_DICT = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    'images_largescale/': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'places', 'sun',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

BENCHMARKS_DICT = {
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(item) or path.endswith(filename):
            return False
    print(f"{filename} needs download:")
    return True


def download_dataset(dataset, data_root):
    # Find the appropriate directory for this dataset
    for key, datasets in DIR_DICT.items():
        if dataset in datasets:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print(f'Invalid dataset detected: {dataset}')
        return

    # Download if needed
    if require_download(dataset, store_path):
        print(f"Downloading to {store_path}")
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        
        # Download the file
        gdown.download(id=DOWNLOAD_ID_DICT[dataset], output=store_path)
        
        # Extract the zip file
        file_path = os.path.join(store_path, f"{dataset}.zip")
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        
        # Clean up
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    # Ensure data_root ends with a slash
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    # Download benchmark image list if needed
    benchmark_path = os.path.join(data_root, 'benchmark_imglist')
    if not os.path.exists(benchmark_path):
        gdown.download(id=DOWNLOAD_ID_DICT['benchmark_imglist'], output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    # Download all required datasets for the benchmark
    for dataset in BENCHMARKS_DICT[id_data_name]:
        download_dataset(dataset, data_root)


class CustomTensorDataset(TensorDataset):
    def __getitem__(self, index):
        return dict(zip(("data", "label"), tuple(tensor[index] for tensor in self.tensors)))


def _load_dataset(data_dir: str, split: str) -> TensorDataset | None:
    split_path = Path(data_dir) / f"{split}.pt"
    try:
        data = torch.load(split_path, weights_only=False)
        return CustomTensorDataset(*data.tensors)
    except FileNotFoundError:
        return None


def _split_dataset(
    base_dataset: Dataset, 
    test_frac: float, 
    stratify: torch.Tensor | None = None
) -> tuple[Subset, Subset]:
    train_idx, test_idx = train_test_split(
        torch.arange(len(base_dataset)),
        stratify=stratify,
        test_size=test_frac,
    )
    return Subset(base_dataset, train_idx), Subset(base_dataset, test_idx)


def _get_preprocessor_stats(preprocessor):
    """Extract mean and std from various preprocessor types."""
    if tvs_new:
        if isinstance(preprocessor, tvs.transforms._presets.ImageClassification):
            return preprocessor.mean, preprocessor.std
        elif isinstance(preprocessor, tvs.transforms.Compose):
            temp = preprocessor.transforms[-1]
            return temp.mean, temp.std
        elif isinstance(preprocessor, BasePreprocessor):
            temp = preprocessor.transform.transforms[-1]
            return temp.mean, temp.std
        else:
            raise TypeError("Unsupported preprocessor type")
    else:
        if isinstance(preprocessor, tvs.transforms.Compose):
            temp = preprocessor.transforms[-1]
            return temp.mean, temp.std
        elif isinstance(preprocessor, BasePreprocessor):
            temp = preprocessor.transform.transforms[-1]
            return temp.mean, temp.std
        else:
            raise TypeError("Unsupported preprocessor type")


def _load_id_datasets(id_name, data_root, dataset_metadata, preprocessor, 
                     test_standard_preprocessor, loader_kwargs):
    """Load in-distribution datasets."""
    sub_dataloader_dict = {}
    for split in dataset_metadata['id'].keys():
        dataset = ImglistDataset(
            name=f"{id_name}_{split}",
            imglist_pth=os.path.join(data_root, dataset_metadata['id'][split]['imglist_path']),
            data_dir=os.path.join(data_root, dataset_metadata['id'][split]['data_dir']),
            num_classes=dataset_metadata['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor
        )
        sub_dataloader_dict[split] = DataLoader(dataset, **loader_kwargs)
    return sub_dataloader_dict


def get_id_ood_dataloader(id_name, data_root, preprocessor, id_preembedded_dir=None, 
                         selected_ood_datasets=None, **loader_kwargs):
    """
    Create dataloaders for in-distribution and out-of-distribution datasets.
    
    Args:
        id_name: Name of the in-distribution dataset
        data_root: Root directory for datasets
        preprocessor: Preprocessor for transforming images
        id_preembedded_dir: Directory with pre-embedded features (optional)
        selected_ood_datasets: List of OOD datasets to include (optional)
        **loader_kwargs: Additional arguments for DataLoader
        
    Returns:
        Dictionary containing dataloaders for ID, CSID, and OOD datasets
    """
    # Initialize dataloaders dictionary
    dataloader_dict = {'id': {}, 'csid': {}, 'ood': {}}
    dataset_metadata = DATASET_METADATA[id_name]
    
    # Get preprocessors
    test_standard_preprocessor = get_default_preprocessor(id_name)
    
    # Special handling for ImageNet datasets
    imagenet_c_preprocessor = None
    if 'imagenet' in id_name:
        mean, std = _get_preprocessor_stats(preprocessor)
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)
    
    # Handle ID datasets
    if id_name == "imagenet" and id_preembedded_dir:
        # Load pre-embedded datasets
        logger.info(f"Loading preembedded dataset from {id_preembedded_dir}")
        train_ds = _load_dataset(data_dir=id_preembedded_dir, split="train")
        val_ds = _load_dataset(data_dir=id_preembedded_dir, split="val")
        test_ds = _load_dataset(data_dir=id_preembedded_dir, split="test") or val_ds
        
        assert train_ds is not None, "Train dataset not found"
        assert val_ds is not None, "Validation dataset not found"
        assert test_ds is not None, "Test dataset not found"
        
        dataloader_dict["id"]["train"] = DataLoader(train_ds, **loader_kwargs)
        dataloader_dict["id"]["val"] = DataLoader(val_ds, **loader_kwargs)
        dataloader_dict["id"]["test"] = DataLoader(test_ds, **loader_kwargs)
    else:
        # Load ID datasets from image lists
        sub_dataloader_dict = _load_id_datasets(id_name, data_root, dataset_metadata, preprocessor, test_standard_preprocessor, loader_kwargs)
        for split, dataloader in sub_dataloader_dict.items():
            dataloader_dict['id'][split] = dataloader
    
    # Handle CSID (Covariate Shift ID) datasets
    for dataset_name in dataset_metadata['csid']['datasets']:
        # Select appropriate preprocessor
        current_preprocessor = (
            imagenet_c_preprocessor if dataset_name == 'imagenet_c' else preprocessor
        )
        
        dataset = ImglistDataset(
            name=f"{id_name}_csid_{dataset_name}",
            imglist_pth=os.path.join(
                data_root, dataset_metadata['csid'][dataset_name]['imglist_path']
            ),
            data_dir=os.path.join(
                data_root, dataset_metadata['csid'][dataset_name]['data_dir']
            ),
            num_classes=dataset_metadata['num_classes'],
            preprocessor=current_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor
        )
        dataloader_dict['csid'][dataset_name] = DataLoader(dataset, **loader_kwargs)
    
    # Handle OOD datasets
    dataloader_dict['ood'] = {'near': {}, 'far': {}}
    
    # Add validation OOD dataset
    val_config = dataset_metadata['ood'].get('val')
    if val_config:
        dataset = ImglistDataset(
            name=f"{id_name}_ood_val",
            imglist_pth=os.path.join(data_root, val_config['imglist_path']),
            data_dir=os.path.join(data_root, val_config['data_dir']),
            num_classes=dataset_metadata['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor
        )
        dataloader_dict['ood']['val'] = DataLoader(dataset, **loader_kwargs)
    
    # Add near/far OOD datasets
    for split in ['near', 'far']:
        if split not in dataset_metadata['ood']:
            continue
            
        split_config = dataset_metadata['ood'][split]
        available_datasets = split_config.get('datasets', [])
        
        # Filter datasets if selected_ood_datasets is provided
        if selected_ood_datasets:
            datasets_to_load = [d for d in available_datasets if d in selected_ood_datasets]
        else:
            datasets_to_load = available_datasets
            
        for dataset_name in datasets_to_load:
            if dataset_name not in split_config:
                logger.warning(f"Dataset {dataset_name} configuration not found")
                continue
                
            dataset_config = split_config[dataset_name]
            dataset = ImglistDataset(
                name=f"{id_name}_ood_{dataset_name}",
                imglist_pth=os.path.join(data_root, dataset_config['imglist_path']),
                data_dir=os.path.join(data_root, dataset_config['data_dir']),
                num_classes=dataset_metadata['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor
            )
            dataloader_dict['ood'][split][dataset_name] = DataLoader(dataset, **loader_kwargs)
    
    return dataloader_dict
