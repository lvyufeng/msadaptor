from functools import partial

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from ..inception import Inception_V3_Weights

__all__ = [
    "QuantizableInception3",
    "Inception_V3_QuantizedWeights",
    "inception_v3",
]

def inception_v3():
    raise NotImplementedError("Quantization model are not currently supported")

class QuantizableInception3(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

class Inception_V3_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth",
        transforms=partial(ImageClassification, crop_size=299, resize_size=342),
        meta={
            "num_params": 27161264,
            "min_size": (75, 75),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "fbgemm",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": Inception_V3_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.176,
                    "acc@5": 93.354,
                }
            },
            "_docs": """
                These weights were produced by doing Post Training Quantization (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1