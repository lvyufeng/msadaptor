"""creation ops"""
from .._bind import get_default_dtype
from mindspore.ops.auto_generate.gen_arg_handler import dtype_to_type_id

import torch
from torch.types import device as device_
from torch.executor import execute

def as_strided(self, size, stride, storage_offset=None):
    return execute('as_strided', self, size, stride, storage_offset)

# from_numpy
def from_numpy(ndarray):
    return torch.Tensor(ndarray)

# frombuffer

# zeros
def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = device_('cpu')
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute('zeros', size, dtype_to_type_id('Zeros', 'type', dtype),
                     device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# zeros_like
def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if device.type == 'cpu':
        return execute('zeros_like', input, device=device, requires_grad=requires_grad, is_leaf=True)
    return execute('zeros_like_ext', input, dtype_to_type_id('ZerosLikeExt', 'dtype', dtype),
                   device=device, requires_grad=requires_grad, is_leaf=True)

# ones
def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = device_('cpu')
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute('ones', size, dtype_to_type_id('Ones', 'type', dtype),
                     device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# ones_like
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if device.type == 'cpu':
        return execute('ones_like', input, device=device, requires_grad=requires_grad, is_leaf=True)
    return execute('ones_like_ext', input, dtype_to_type_id('OnesLikeExt', 'dtype', dtype),
                   device=device, requires_grad=requires_grad, is_leaf=True)

# arange
def arange(start=0, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    if dtype is None:
        dtype = torch.int64
    if device is None:
        device = device_('cpu')
    if device.type == 'cpu':
        output = execute('range', start, end, step, 1000000,
                         device=device, requires_grad=requires_grad, is_leaf=True)
    else:
        output = execute('arange', start, end, step, dtype_to_type_id('Arange', 'dtype', dtype),
                         device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# range
def range(start=0, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        raise TypeError('range() missing 1 required positional arguments: "end"')
    if dtype is None:
        dtype = torch.int64
    if device is None:
        device = device_('cpu')
    output = execute('range', start, end + 1, step, 1000000,
                     device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# linspace
def linspace(start, end, steps, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = device_('cpu')
    if device.type == 'cpu':
        start = torch.tensor(start, device=device, dtype=dtype)
        end = torch.tensor(end, device=device, dtype=dtype)
        output = execute('linspace', start, end, steps,
                         device=device, requires_grad=requires_grad, is_leaf=True)
    else:
        output = execute('lin_space_ext', start, end, steps, dtype_to_type_id('LinSpaceExt', 'dtype', dtype),
                         device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# logspace

# eye
def eye(n, m=None, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if device is None:
        device = device_('cpu')
    if dtype is None:
        dtype = get_default_dtype()
    output = execute('eye', n, m, dtype_to_type_id('Eye', 'dtype', dtype),
                     device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# empty
def empty(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, memory_format=None):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = device_('cpu')
    output = execute('empty', size, dtype, device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# empty_like
def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return empty(input.shape, dtype=input.dtype, layout=layout, device=input.device, requires_grad=requires_grad)

# empty_strided


# full
def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = device_('cpu')
    if device.type == 'cpu':
        if not isinstance(fill_value, torch.Tensor):
            fill_value = torch.tensor(fill_value, dtype=dtype, device=device)
        output = execute('full', size, fill_value, device=device, requires_grad=requires_grad, is_leaf=True)
    else:
        if isinstance(fill_value, torch.Tensor):
            output = execute('fill_tensor', size, fill_value, dtype_to_type_id('FillScalar', 'dtype', dtype),
                             device=device, requires_grad=requires_grad, is_leaf=True)
        else:
            output = execute('fill_scalar', size, fill_value, dtype_to_type_id('FillTensor', 'dtype', dtype),
                             device=device, requires_grad=requires_grad, is_leaf=True)
    if out is None:
        return output
    out.data = output
    return out

# full_like
def full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return full(input.shape, fill_value, dtype=dtype, layout=layout, device=input.device, requires_grad=requires_grad)

# quantize_per_tensor


# quantize_per_channel


# dequantize


# complex


# polar
def polar(abs, angle, *, out=None):
    output = execute('polar', abs, angle)
    if out is None:
        return output
    out.data = output
    return out


# heaviside

__all__ = ['arange', 'as_strided', 'empty', 'empty_like',
           'eye', 'from_numpy', 'full', 'full_like',
           'linspace', 'ones', 'ones_like',
           'polar', 'range', 'zeros', 'zeros_like'
]
