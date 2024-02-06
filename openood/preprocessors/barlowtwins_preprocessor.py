import abc

import torch.nn
import torchvision.transforms as T
from PIL.Image import Image
from lightly.transforms import BYOLView1Transform, BYOLView2Transform
from lightly.transforms.multi_view_transform import MultiViewTransform
from typing import Optional

from openood.preprocessors import BasePreprocessor
from openood.preprocessors.transform import normalization_dict
from openood.utils.config import Config


class BarlowTwinsPreprocessor(BasePreprocessor):
    def __init__(self, config: Config):
        # super().__init__(config)
        self.image_size = config.dataset.img_size
        self.interpolation = T.InterpolationMode.BILINEAR
        normalization_type = config.dataset.normalization_type
        if normalization_type in normalization_dict.keys():
            self.mean = normalization_dict[normalization_type][0]
            self.std = normalization_dict[normalization_type][1]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]


        self.transform = BarlowTwinsTransform(
                view_1_transform=BYOLView1Transform(
                    input_size=self.image_size, gaussian_blur=0.0, normalize=self.normalize
                ),
                view_2_transform=BYOLView2Transform(
                    input_size=self.image_size, gaussian_blur=0.0, normalize=self.normalize
                ),
                view_3_transform=FinetuneViewTransform(
                    input_size=self.image_size, normalize=self.normalize
                ),
            )

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)

    @property
    def normalize(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class DatasetTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SelfSupervisedTransform(DatasetTransform):
    def __init__(
        self,
        train: bool = True,
        input_height: int = 32,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize: Optional[torch.nn.Module] = None,
    ):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = T.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(T.RandomApply([T.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = T.Compose(color_transform)

        if normalize is None:
            self.final_transform = T.ToTensor()
        else:
            self.final_transform = T.Compose(
                [
                    T.Resize((self.input_height, self.input_height)),
                    T.ToTensor(),
                    normalize,
                ]
            )

        self.transform = T.Compose(
            [
                T.RandomResizedCrop(self.input_height),
                T.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = self.final_transform

    def __call__(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert callable(self.transform) and callable(self.finetune_transform)
        return self.transform(sample), self.transform(sample), self.finetune_transform(sample)


class SupervisedTransform(DatasetTransform):
    def __init__(
        self,
        train: bool = True,
        input_height: int = 32,
        normalize: Optional[dict] = None,
    ):
        if train:
            transform = [
                T.RandomResizedCrop(input_height),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        else:
            transform = [T.Resize(input_height), T.ToTensor()]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        self.transform = T.Compose(transform)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        assert callable(self.transform)
        return self.transform(sample)


class ImageNetSupervisedTransform(DatasetTransform):
    def __init__(
        self,
        train: bool = True,
        resize_height: int = 256,
        input_height: int = 224,
        normalize: Optional[dict] = None,
    ):
        if train:
            transform = [
                T.RandomResizedCrop(input_height),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        else:
            transform = [T.Resize(resize_height), T.CenterCrop(input_height), T.ToTensor()]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        self.transform = T.Compose(transform)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        assert callable(self.transform)
        return self.transform(sample)


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class FinetuneViewTransform(DatasetTransform):
    def __init__(self, input_size: int = 32, normalize: Optional[dict] = None):
        self.transform = T.Compose([
            Convert('RGB'),
            T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ])

    def __call__(self, image: torch.Tensor | Image) -> torch.Tensor:
        return self.transform(image)


class BarlowTwinsTransform(MultiViewTransform, DatasetTransform):
    """Adapted from https://github.com/lightly-ai/lightly/blob/master/lightly/transforms/byol_transform.py"""

    def __init__(
        self,
        view_1_transform: BYOLView1Transform | None = None,
        view_2_transform: BYOLView2Transform | None = None,
        view_3_transform: Optional[FinetuneViewTransform] = None,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or BYOLView1Transform()
        view_2_transform = view_2_transform or BYOLView2Transform()
        view_3_transform = view_3_transform or FinetuneViewTransform()
        super().__init__(transforms=[view_1_transform, view_2_transform, view_3_transform])
