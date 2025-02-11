import numpy as np
import torch

__all__ = []

def randn_cpu(*args):
    data = np.random.randn(*args[0])
    return data

__all__.append('randn_cpu')