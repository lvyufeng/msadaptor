from functools import partial

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from ..mobilenetv3 import MobileNet_V3_Large_Weights

__all__ = [
    "QuantizableMobileNetV3",
    "MobileNet_V3_Large_QuantizedWeights",
    "mobilenet_v3_large",
]

class QuantizableMobileNetV3(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

def mobilenet_v3_large():
    raise NotImplementedError("Quantization model are not currently supported")

class MobileNet_V3_Large_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 5483032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "qnnpack",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv3",
            "unquantized": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.004,
                    "acc@5": 90.858,
                }
            },
            "_docs": """
                These weights were produced by doing Quantization Aware Training (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1