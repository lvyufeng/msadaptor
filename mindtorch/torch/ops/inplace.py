import torch
from torch.executor import execute

def inplace_copy(self, other):
    if self.device != other.device:
        other = other.to(self.device)
    if self.device.type == 'cpu':
        # execute('assign', self, other)
        # # self._data.assign_value_cpp(other._data)
        self.data = other
    else:
        execute('inplace_copy', self, other)
    return self

def inplace_zero(input):
    execute('inplace_zero', input)

def inplace_fill(input, value):
    if isinstance(value, (int, float, bool)):
        execute('inplace_fill_scalar', input, value)
    execute('inplace_fill_tensor', input, value)


__all__ = [
    'inplace_copy',
    'inplace_zero',
]
