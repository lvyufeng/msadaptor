"""array op"""
import numbers
from mindspore.ops.auto_generate.gen_arg_handler import str_to_enum
from mindspore._c_expression import Tensor as MSTensor
import numpy as np

import torch
from torch.executor import execute

def t(input):
    assert input.ndim <= 2
    if input.ndim == 2:
        return transpose(input, 0, 1)
    return input

# adjoint

# argwhere
def argwhere(input):
    return execute('nonzero', input)

# cat
def cat(tensors, dim=0):
    return execute('concat', tensors, dim)

# concat
def concat(tensors, dim=0):
    return cat(tensors, dim)

# concatenate
def concatenate(tensors, dim=0):
    return cat(tensors, dim)

# conj
def conj(input):
    return execute('conj', input)

# chunk
def chunk(input, chunks, dim=0):
    return execute('chunk', input, chunks, dim)

# dsplit


# column_stack


# dstack


# gather
def gather(input, dim, index):
    return execute('gather_d', input, dim, index)

def gather_nd(input, indices):
    return execute('gather_nd', input, indices)

# hsplit


# hstack

# index_fill


# index_add
def index_add(input, dim, index, source, *, alpha=1):
    return execute('index_add_ext', input, index, source, dim, alpha)

# index_copy


# index_reduce


# index_select
def index_select(input, dim, index):
    return execute('index_select', input, dim, index)

# masked_select
def masked_select(input, mask):
    return execute('masked_select', input, mask)

# movedim


# moveaxis


# narrow
def narrow(input, dim, start, length):
    return execute('narrow', input, dim, start, length)

# narrow_copy


# nonzero
def nonzero(input, *, as_tuple=False):
    if as_tuple:
        return execute('non_zero_ext', input)
    return execute('non_zero', input)

# permute
def permute(input, dims):
    assert isinstance(dims, tuple)
    return execute('transpose', input, *dims)

# reshape
def reshape(input, *shape):
    if isinstance(shape[0], tuple):
        shape = shape[0]
    return execute('reshape', input, shape)

def view(input, *shape):
    return reshape(input, shape)

# row_stack

# select
def select(input, dim, index):
    return execute('select_ext', input, dim, index)

# scatter
def scatter(input, dim, index, src):
    return execute('scatter', input, dim, index, src, str_to_enum('Scatter', 'reduce', 'none'))

# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
def scatter_add(input, dim, index, src):
    return execute('scatter_add_ext', input, dim, index, src)

# scatter_reduce


# split
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        res = execute('split_tensor', tensor, split_size_or_sections, dim)
    elif isinstance(split_size_or_sections, (list, tuple)):
        res = execute('split_with_size', tensor, split_size_or_sections, dim)
    else:
        raise TypeError(f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), "
                        f"but got {type(split_size_or_sections)}")
    return res

# squeeze
def squeeze(input, dim=None):
    return execute('squeeze', input, dim)

# stack

def stack(tensors, dim=0):
    if tensors[0].device.type == 'cpu':
        return execute('stack', tensors, dim)
    return execute('stack_ext', tensors, dim)

# swapaxes
def swapaxes(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# swapdims
def swapdims(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# take
def take(input, index):
    input = input.view(-1)
    index_shape = index.shape
    index = index.view(-1)
    return gather(input, 0, index).view(index_shape)

# take_along_dim


# tensor_split


# tile
def tile(input, dims):
    return execute('tile', input, dims)

# transpose
def transpose(input, dim0, dim1):
    return execute('transpose_ext', input, dim0, dim1)

# unbind
def unbind(input, dim=0):
    return execute('unstack_ext', input, dim)

# unravel_index

# unsqueeze
def unsqueeze(input, dim):
    return execute('expand_dims', input, dim)

# vsplit

# vstack

# where
def where(condition, input, other):
    return execute('select', condition, input, other)


tensor_1d = MSTensor([0], dtype=torch.int64)
empty_tensor_1d = MSTensor(shape=(0,), dtype=torch.int64)
empty_tensor_9d = MSTensor(shape=(0,)*9, dtype=torch.int64)

def _do_select(self, dim: int, index: int, dim_index: int, self_shape: list):
    """call select view operator"""
    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    dim_size = self_shape[dim]
    if index >= dim_size or index < -dim_size:
        raise IndexError(f"Index {index} is out of bounds for dimension {dim_index} with size {dim_size}")
    index = index + dim_size if index < 0 else index
    return execute('select_ext', self, dim, index)


def _do_slice(self, dim: int, index: slice, self_shape: list):
    """call slice view operator"""
    def _get_index(index, default):
        if index is None:
            return default
        return index

    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    step = _get_index(index.step, 1)
    if step <= 0:
        raise ValueError("slice step must be positive")
    start = _get_index(index.start, 0)
    end = _get_index(index.stop, self_shape[dim])
    if start == 0 and end == self_shape[dim] and step == 1:
        return self
    return execute('slice_ext', self, dim, start, end, step)

def _wrap_index_to_tuple(index):
    """Wrap index to tuple"""
    if isinstance(index, tuple):
        return index
    if isinstance(index, list):
        if len(index) < 32 and any(isinstance(i, (torch.Tensor, list, tuple, slice, type(None), Ellipsis)) for i in index):
            return tuple(index)
    return (index,)


def _count_indexed_dims(indexes):
    """Count indexed dims"""
    count = 0
    for index in indexes:
        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                count += index.ndim
            else:
                count += 1
        elif not isinstance(index, (type(None), type(...), bool)):
            count += 1
    return count

def tensor_getitem(self, index):
    """Handle tensor getitem"""
    if isinstance(index, bool):
        self_viewed = unsqueeze(self, 0)
        index_for_bool = tensor_1d if index else empty_tensor_1d
        return execute('index', self_viewed, [index_for_bool])
    if isinstance(index, int):
        return _do_select(self, 0, index, 0, list(self.shape))
    if isinstance(index, slice):
        result = _do_slice(self, 0, index, list(self.shape))
        return result
    if index is None:
        return unsqueeze(self, 0)
    if isinstance(index, Ellipsis):
        return self
    indexes = _wrap_index_to_tuple(index)
    indexed_dims = _count_indexed_dims(indexes)
    if self.ndim < indexed_dims:
        raise IndexError(f"too many indices for tensor with dimension size {self.ndim}")
    remain_indexes = []
    self_viewed, remain_indexes = _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims)
    if not remain_indexes:
        return self_viewed
    return execute('index', self_viewed, remain_indexes)


def tensor_setitem(self, index, value):
    """Handle tensor setitem"""
    if not isinstance(value, torch.Tensor):
        if isinstance(value, (bool, int, float)):
            value = torch.Tensor(value, device=self.device)
        else:
            raise TypeError(f"Can't assign a {type(value)} to a {self.dtype}.")

    if isinstance(index, bool) and index is False:
        return self
    if isinstance(index, type(...)):
        execute('inplace_copy', self, value)
        return self
    if index is None or (isinstance(index, bool) and index is True):
        self_viewed = unsqueeze(self, 0)
        execute('inplace_copy', self_viewed, value)
        return self
    if isinstance(index, int):
        self_viewed = _do_select(self, 0, index, 0, list(self.shape))
        execute('inplace_copy', self_viewed, value)
        return self
    if isinstance(index, slice):
        self_viewed = _do_slice(self, 0, index, list(self.shape))
        execute('inplace_copy', self_viewed, value)
        return self
    indexes = _wrap_index_to_tuple(index)
    indexed_dims = _count_indexed_dims(indexes)
    if self.ndim < indexed_dims:
        raise IndexError(f"too many indices for tensor with dimension size {self.ndim}")
    remain_indexes = []
    self_viewed, remain_indexes = _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims)
    if not remain_indexes:
        execute('inplace_copy', self_viewed, value)
        return self
    execute('inplace_index_put', self_viewed, remain_indexes, value)
    return self

__all__ = [
    # adjoint,
    'argwhere',
    'cat',
    'concat',
    'concatenate',
    'conj',
    'chunk',
    # dsplit,
    # column_stack
    # dstack
    'gather',
    'gather_nd',
    # hsplit
    'index_add',
    # index_copy
    # index_reduce
    'index_select',
    'masked_select',
    # movedim
    # moveaxis
    'narrow',
    # narrow_copy
    'nonzero',
    'permute',
    'reshape',
    'view',
    # row_stack
    'select',
    'scatter',
    # diagonal_scatter
    # select_scatter
    # slice_scatter
    'scatter_add',
    # scatter_reduce
    'split',
    'squeeze',
    'stack',
    'swapaxes',
    'swapdims',
    'take',
    # take_along_dim
    # tensor_split
    'tile',
    'transpose',
    'unbind',
    # unravel_index
    'unsqueeze',
    # vsplit
    'where',
    'tensor_getitem',
    'tensor_setitem',
    't'
]
