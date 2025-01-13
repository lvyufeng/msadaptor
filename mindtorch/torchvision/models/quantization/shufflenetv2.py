from functools import partial

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from ..shufflenetv2 import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X1_5_Weights,
    ShuffleNet_V2_X2_0_Weights,
)

__all__ = [
    "QuantizableShuffleNetV2",
    "ShuffleNet_V2_X0_5_QuantizedWeights",
    "ShuffleNet_V2_X1_0_QuantizedWeights",
    "ShuffleNet_V2_X1_5_QuantizedWeights",
    "ShuffleNet_V2_X2_0_QuantizedWeights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


class QuantizableShuffleNetV2(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

def shufflenet_v2_x0_5():
    raise NotImplementedError("Quantization model are not currently supported")

def shufflenet_v2_x1_0():
    raise NotImplementedError("Quantization model are not currently supported")

def shufflenet_v2_x1_5():
    raise NotImplementedError("Quantization model are not currently supported")

def shufflenet_v2_x2_0():
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


class ShuffleNet_V2_X0_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x0.5_fbgemm-00845098.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 1366792,
            "unquantized": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 57.972,
                    "acc@5": 79.780,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1


class ShuffleNet_V2_X1_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 2278604,
            "unquantized": ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 68.360,
                    "acc@5": 87.582,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1


class ShuffleNet_V2_X1_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x1_5_fbgemm-d7401f05.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/5906",
            "num_params": 3503624,
            "unquantized": ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.052,
                    "acc@5": 90.700,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1


class ShuffleNet_V2_X2_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x2_0_fbgemm-5cac526c.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/5906",
            "num_params": 7393996,
            "unquantized": ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.354,
                    "acc@5": 92.488,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1