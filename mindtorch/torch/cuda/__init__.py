import threading

from mindspore import Tensor
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.hal import *

FloatTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor

def manual_seed_all(seed: int):
    manual_seed(seed)

def current_device():
    return -1

def is_available():
    return True

def set_device(device):
    pass

def _lazy_call(callable, **kwargs):
    callable()
