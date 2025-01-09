"""array op"""
from mindspore.ops.auto_generate.gen_arg_handler import str_to_enum

from torch.executor import execute

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
    return execute('transpose', input, dims)

# reshape
def reshape(input, shape):
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
]
