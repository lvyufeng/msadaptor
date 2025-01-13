from functools import partial

from ...transforms._presets import ImageClassification
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from ..mobilenetv2 import MobileNet_V2_Weights

__all__ = [
    "QuantizableMobileNetV2",
    "MobileNet_V2_QuantizedWeights",
    "mobilenet_v2",
]

class QuantizableMobileNetV2(object):
    def __init__(self):
        raise NotImplementedError("Quantization model are not currently supported")

def mobilenet_v2():
    raise NotImplementedError("Quantization model are not currently supported")

class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 3504872,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "backend": "qnnpack",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNet_V2_Weights.IMAGENET1K_V1,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.658,
                    "acc@5": 90.150,
                }
            },
            "_docs": """
                These weights were produced by doing Quantization Aware Training (eager mode) on top of the unquantized
                weights listed below.
            """,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1