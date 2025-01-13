from functools import partial

from ..resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
)

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES

__all__ = [
    "QuantizableResNet",
    "ResNet18_QuantizedWeights",
    "ResNet50_QuantizedWeights",
    "ResNeXt101_32X8D_QuantizedWeights",
    "ResNeXt101_64X4D_QuantizedWeights",
    "resnet18",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
]

class QuantizableResNet(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

def resnet18():
    raise NotImplementedError("Quantization model are not currently supported")

def resnet50():
    raise NotImplementedError("Quantization model are not currently supported")

def resnext101_32x8d():
    raise NotImplementedError("Quantization model are not currently supported")

def resnext101_64x4d():
    raise NotImplementedError("Quantization model are not currently supported")

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
    "backend": "fbgemm",
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
    "_docs": """
        These weights were produced by doing Post Training Quantization (eager mode) on top of the unquantized
        weights listed below.
    """,
}

class ResNet18_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "unquantized": ResNet18_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.494,
                    "acc@5": 88.882,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1

class ResNet50_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "unquantized": ResNet50_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.920,
                    "acc@5": 92.814,
                }
            },
        },
    )
    IMAGENET1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm-23753f79.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "unquantized": ResNet50_Weights.IMAGENET1K_V2,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.282,
                    "acc@5": 94.976,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V2

class ResNeXt101_32X8D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "unquantized": ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.986,
                    "acc@5": 94.480,
                }
            },
        },
    )
    IMAGENET1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm-ee16d00c.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "unquantized": ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.574,
                    "acc@5": 96.132,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V2

class ResNeXt101_64X4D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_64x4d_fbgemm-605a1cb3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 83455272,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "unquantized": ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.898,
                    "acc@5": 96.326,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1