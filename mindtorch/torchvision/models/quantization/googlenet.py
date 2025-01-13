from functools import partial

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from ..googlenet import GoogLeNet_Weights

__all__ = [
    "QuantizableGoogLeNet",
    "GoogLeNet_QuantizedWeights",
    "googlenet",
]

def googlenet():
    raise NotImplementedError("Quantization model are not currently supported")

class QuantizableGoogLeNet(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

class GoogLeNet_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 6624904,
            "min_size": (15, 15),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "fbgemm",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": GoogLeNet_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.826,
                    "acc@5": 89.404,
                }
            },
            "_docs": """
                These weights were produced by doing Post Training Quantization (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1