"""reduction op"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, DEVICE_TARGET
from mindspore.ops.auto_generate.gen_arg_handler import dtype_to_type_id

import torch
from torch.executor import execute

# argmax
def argmax(input, dim=None, keepdim=False):
    return execute('argmax_ext', input, dim, keepdim)

# argmin
def argmin(input, dim=None, keepdim=False):
    return execute('argmin_ext', input, dim, keepdim)

# amax
def amax(input, dim, keepdim=False):
    return execute('reduce_max', input, dim, keepdim)

# amin
def amin(input, dim, keepdim=False):
    return execute('reduce_min', input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
def all(input, dim=None, keepdim=False, *, dtype=None):
    return execute('reduce_all', input, dim, keepdim)

# any
def any(input, dim=None, keepdim=False):
    return execute('reduce_any', input, dim, keepdim)

# max
def max(input, dim=None, keepdim=False):
    return execute('max_dim', input, dim, keepdim)

# min
def min(input, dim=None, keepdim=False):
    return execute('min_dim', input, dim, keepdim)

# dist

# logsumexp
def logsumexp(input, dim, keepdim=False):
    return execute('logsumexp', input, dim, keepdim)

# mean
def mean(input, dim=None, keepdim=False, *, dtype=None):
    return execute('mean_ext', input, dim, keepdim,
                   dtype if dtype is None dtype_to_type_id('MeanExt', 'dtype', dtype))

# nanmean


# median
def median(input, dim=-1, keepdim=False):
    if dim is None:
        return execute('median_ext', input)
    return execute('median_dim', input, dim, keepdim)

# nanmedian


# mode


# norm
def norm(input, p='fro', dim=None, keepdim=False, dtype=None):
    if use_pyboost() and has_norm:
        return mindspore.mint.norm(input, p, dim, keepdim, dtype=dtype)
    return ops.norm(input, p, dim, keepdim, dtype=dtype)

# nansum
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    return execute('nansum', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('Nansum', 'dtype', dtype))

# prod
def prod(input, dim=None, keepdim=False, *, dtype=None):
    return execute('prod_ext', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('ProdExt', 'dtype', dtype))

# quantile

# nanquantile

# std
def std(input, dim=None, *, correction=1, keepdim=False):
    return execute('std', input, dim, correction, keepdim)

# std_mean
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    return execute('std_mean', input, dim, correction, keepdim)

# sum
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if 0 in input.shape:
        return torch.tensor(0, dtype=dtype, device=input.device)
    return execute('sum_ext', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('SumExt', 'dtype', dtype))

# unique
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique:
        return mindspore.mint.unique(input, sorted, return_inverse, return_counts, dim)

    out, inverse = ops.unique(input)
    outs = (out,)
    if return_inverse:
        outs += (inverse,)
    if return_counts:
        counts = (out == input).sum(0, keepdims=True)
        outs += (counts,)
    return outs if len(outs) > 1 else outs[0]

# unique_consecutive
has_unique_consecutive = hasattr(mindspore.mint, 'unique_consecutive')
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique_consecutive:
        return mindspore.mint.unique_consecutive(input, return_inverse, return_counts, dim)
    return ops.unique_consecutive(input, return_inverse, return_counts, dim)

# var
has_var = hasattr(mindspore.mint, 'var')
def var(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_var:
        return mindspore.mint.var(input, dim=dim, correction=correction, keepdim=keepdim)
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2)

# var_mean
has_var_mean = hasattr(mindspore.mint, 'var_mean')
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_var_mean:
        return mindspore.mint.var_mean(input, dim=dim, correction=correction, keepdim=keepdim)
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2), \
        mean(input, dim, keepdim)

# count_nonzero
has_count_nonzero = hasattr(mindspore.mint, 'count_nonzero')
def count_nonzero(input, dim=None):
    if use_pyboost() and has_count_nonzero:
        return mindspore.mint.count_nonzero(input, dim)
    if dim is None:
        dim = ()
    return ops.count_nonzero(input, dim)

__all__ = ['all', 'amax', 'amin', 'aminmax', 'any', 'argmax', 'argmin', 'count_nonzero', 'logsumexp', 'max', 'mean', 'median', 'min', 'nanmedian', 'nanquantile', 'nansum', 'norm', 'prod', 'quantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive', 'var', 'var_mean']
 