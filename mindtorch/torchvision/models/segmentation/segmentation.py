import warnings

# Import all methods/classes for BC:
from . import *


warnings.warn(
    "The 'torchvision.models.segmentation.segmentation' module is deprecated since 0.2.0 and will be removed in "
    "0.3.0. Please use the 'torchvision.models.segmentation' directly instead."
)
