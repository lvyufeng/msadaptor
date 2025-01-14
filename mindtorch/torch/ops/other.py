"""other op"""
import copy
import numpy as np
import mindspore
from mindspore.ops.auto_generate.gen_arg_handler import dtype_to_type_id

from .reduction import any
from .comparison import eq

import torch
from torch.executor import execute

# atleast_2d


# atleast_3d


# bincount
def bincount(input, weights=None, minlength=0):
    return execute('bincount_ext', input, weights, minlength)

# block_diag


# broadcast_tensors
def broadcast_tensors(*tensors):
    target_shape = broadcast_shapes(*[t.shape for t in tensors])
    broadcasted_tensors = [t.broadcast_to(target_shape) for t in tensors]
    return broadcasted_tensors


# broadcast_to
def broadcast_to(input, shape):
    return execute('broadcast_to', input, shape)


# broadcast_shapes
def broadcast_shapes(*shapes):
    reversed_shapes = [list(reversed(shape)) for shape in shapes]

    max_dim = max(len(shape) for shape in reversed_shapes)

    result_shape = [1] * max_dim

    for i in range(max_dim):
        current_dim_size = 1
        for shape in reversed_shapes:
            if i < len(shape):
                if shape[i] == 1:
                    continue
                if current_dim_size == 1:
                    current_dim_size = shape[i]
                elif current_dim_size != shape[i]:
                    raise ValueError(f"Shapes {shapes} are not broadcastable.")
        result_shape[i] = current_dim_size

    return tuple(reversed(result_shape))

# bucketize

# cartesian_prod


# cdist
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    return execute('cdist', x1, x2, p)

# clone
def clone(input):
    copy_tensor = copy.deepcopy(input)
    copy_tensor.data = execute('clone', input)
    return copy_tensor


# combinations


# corrcoef


# cov


# cross

# cummax

# cummin

# cumprod

# cumsum
def cumsum(input, dim, dtype=None):
    return execute('cumsum_ext', input, dim,
                   dtype if dtype is None else dtype_to_type_id('CumsumExt', 'dtype', dtype))

# diag
def diag(input):
    return execute('diag', input)


# diag_embed


# diagflat


# diagonal
def diagonal(input, offset=0, dim1=0, dim2=1):
    return execute('diagonal', input, offset, dim1, dim2)

# diff


# einsum


def einsum_label_to_index(label):
    """
    Args:
        label (str): The label representing a dimension in an Einstein sum.
            It should be a single character from the alphabet (upper or lower case) or '.'.

    Returns:
        NoneType: This function returns None.

    Raises:
        None.
    """
    if label == ".":
        return 52
    NUM_OF_LETTERS = ord("z") - ord("a") + 1
    return (
        (ord(label) - ord("A"))
        if (label.isupper())
        else (NUM_OF_LETTERS + (ord(label) - ord("a")))
    )


def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    r"""
    This function takes three parameters: dim, dim_post_expr, and wrap_scalar.

    Args:
        - dim (int): Represents the dimension to be wrapped.
        - dim_post_expr (int): Represents the value used to wrap the dimension.
        - wrap_scalar (bool, optional): Specifies whether a scalar value should be wrapped. Default is True.

    Returns:
        None: This function does not return a value directly.

    Raises:
        AssertionError: Raised if the value of dim_post_expr is less than or equal to 0 and wrap_scalar is False.
        AssertionError: Raised if the value of dim is less than the minimum or greater than the maximum allowed range.
        AssertionError: Raised if the value of dim is negative and cannot be wrapped due to invalid dim_post_expr.

    """
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max)
    if dim < 0:
        dim += dim_post_expr
    return dim


def dim_list_to_bitset(opt_dims, ndims):
    r"""
    Converts a list of optional dimensions to a bitset representation.

    Args:
        opt_dims (List[int]): The list of optional dimensions to be converted to a bitset representation.
        ndims (int): The total number of dimensions.

    Returns:
        List[bool]: A list representing the bitset, where True indicates the presence of the dimension and False indicates its absence.

    Raises:
        None
    """
    if opt_dims:
        seen = [False] * (max(opt_dims) + 1)
        for dim in opt_dims:
            dim = maybe_wrap_dim(dim, ndims)
            seen[dim] = True
    else:
        seen = [True for _ in range(ndims)]
    return seen


def sumproduct_pair(left_, right_, sum_dims_, keep_dim_):
    """
    Calculate the sum-product pair of two arrays along specified dimensions.

    Args:
        left_ (array): The left input array.
        right_ (array): The right input array.
        sum_dims_ (list): A list of dimensions along which to calculate the sum-product pair.
        keep_dim_ (bool): A flag indicating whether to keep the dimensions in the result.

    Returns:
        None. The function performs the sum-product pair calculation and returns None.

    Raises:
        AssertionError: If the number of dimensions of the input arrays do not match,
                       or if non-broadcast dimensions do not match.
    """
    assert left_.ndim == right_.ndim, "number of dimensions must match"
    if len(sum_dims_) == 0:
        return torch.mul(left_, right_)

    dim = left_.ndim
    sum_dims = dim_list_to_bitset(sum_dims_, dim)

    lro, lo, ro = [], [], []
    lro_size, lo_size, ro_size, sum_size = 1, 1, 1, 1
    left = left_
    right = right_

    for i in range(dim):
        sl = left.shape[i] > 1
        sr = right.shape[i] > 1
        if sum_dims[i]:
            if sl and sr:
                assert (
                    left.shape[i] == right.shape[i]
                ), "non-broadcast dimensions must match"
                sum_size *= left.shape[i]
            elif sl:
                left = torch.sum(left, i, keepdim=True)
            elif sr:
                right = torch.sum(right, i, keepdim=True)
        elif sl and sr:
            assert (
                left.shape[i] == right.shape[i]
            ), "non-broadcast dimensions must match"
            lro.append(i)
            lro_size *= left.shape[i]
        elif sl:
            lo.append(i)
            lo_size *= left.shape[i]
        else:
            ro.append(i)
            ro_size *= right.shape[i]

    out_size = []
    for d in lro:
        out_size.append(left.shape[d])
    for d in lo:
        out_size.append(left.shape[d])
    for d in sum_dims_:
        out_size.append(1)
    for d in ro:
        out_size.append(right.shape[d])

    lpermutation = lro.copy()
    lpermutation += lo
    lpermutation += sum_dims_
    lpermutation += ro

    rpermutation = lro.copy()
    rpermutation += sum_dims_
    rpermutation += ro
    rpermutation += lo

    opermutation = [-1] * (len(lro) + len(lo) + len(sum_dims_) + len(ro))
    i = 0
    for it in lro:
        opermutation[it] = i
        i += 1
    for it in lo:
        opermutation[it] = i
        i += 1
    for it in sum_dims_:
        opermutation[it] = i
        i += 1
    for it in ro:
        opermutation[it] = i
        i += 1

    left = torch.permute(left, tuple(lpermutation)).reshape(lro_size, lo_size, sum_size)
    right = torch.permute(right, tuple(rpermutation)).view(lro_size, sum_size, ro_size)

    result = torch.bmm(left, right)
    result = result.view(*out_size).transpose(*opermutation)

    if not keep_dim_:
        sizes = list(result.shape)
        for i in range(dim - 1, 0, -1):
            if sum_dims[i]:
                sizes.pop(i)
        result = result.view(*sizes)

    return result


ELLIPSIS = 52


def einsum(equation, *operands):
    """
    Args:
        equation (str): A string representing the Einstein summation equation to be computed.
            The equation should follow the Einstein summation convention with subscripts in [a-zA-Z],
            commas separating operands, and '->' indicating the output structure.
            It must include at least one operand. An ellipsis '...' can be used to represent multiple dimensions.

    Returns:
        None: This function does not return a value.

    Raises:
        AssertionError: If the function is called without providing at least one operand.
        AssertionError: If an invalid subscript is given in the equation string.
        AssertionError: If the number of subscripts in the equation does not match the number of dimensions for an operand.
        AssertionError: If fewer operands are provided than specified in the equation.
        AssertionError: If more operands are provided than specified in the equation.
        RuntimeError: If operands do not broadcast with remapped shapes [original->remapped].
    """
    assert operands, "einsum(): must provide at least one operand"
    if isinstance(operands[0], tuple):
        operands = operands[0]

    arrow_pos = equation.find("->")
    num_ops = len(operands)
    op_labels = [[] for _ in range(num_ops)]
    lhs = equation[0:arrow_pos]

    curr_op = 0
    found_ell = False
    ell_skip = 0
    for i, label in enumerate(lhs):
        if label == " ":
            continue
        if label == ".":
            if ell_skip != 0:
                ell_skip -= 1
                continue
            assert (
                not found_ell
            ), f"einsum(): found {curr_op} for operand for which an ellipsis was already found"
            assert (
                i + 2 < len(lhs) and lhs[i + 1] == "."
            ), f"einsum(): found {curr_op} for operand that is not part of any ellipsis"
            ell_skip = 2
            op_labels[curr_op].append(ELLIPSIS)
            found_ell = True
        elif label == ",":
            curr_op += 1
            assert (
                curr_op < num_ops
            ), "einsum(): fewer operands were provided than specified in the equation"
            found_ell = False
        else:
            assert str.isalpha(
                label
            ), f"einsum(): invalid subscript given at index {i} in the equation string, subscripts must be in [a-zA-Z]"
            op_labels[curr_op].append(einsum_label_to_index(label))

    assert (
        curr_op == num_ops - 1
    ), "einsum(): more operands were provided than specified in the equation"
    # Labels must be within [a-zA-Z].
    TOTAL_LABELS = 52
    label_count = [0] * TOTAL_LABELS
    # The maximum number of dimensions covered by any ellipsis, needed when
    # unsqueezing missing dimensions from operands to permute and broadcast
    ell_num_dim = 0

    # Compute label frequency and number of dimensions covered by ellipsis
    # We do this after parsing labels to make it more readable and simpler
    # to compute the number of dimensions covered by ellipsis.
    for i, operand in enumerate(operands):
        labels = op_labels[i]
        ndims = operand.ndim
        nlabels = len(labels)
        has_ellipsis = False

        for label in labels:
            if label == ELLIPSIS:
                nlabels -= 1
                has_ellipsis = True
                ell_num_dim = max(ell_num_dim, ndims - nlabels)
            else:
                label_count[label] += 1
        if has_ellipsis:
            assert nlabels <= ndims, (
                f"einsum(): the number of subscripts in the equation ({nlabels}"
                f") is more than the number of dimensions ({ndims}) for operand {i}"
            )
        else:
            assert nlabels == ndims, (
                f"einsum(): the number of subscripts in the equation ({nlabels}"
                f") does not match the number of dimensions ("
                f"{ndims}) for operand {i} and no ellipsis was given"
            )

    # We want to align the dimensions of every input tensor to have
    # shape out_dims + sum_dims. For this, we create a mapping of label
    # to index into the permuted shape.
    label_perm_index = [-1] * TOTAL_LABELS
    # Current index in the permuted shape
    perm_index = 0
    # Start index of ellipsis dimensions in the permuted shape
    ell_index = 0
    found_ell = False

    if arrow_pos == -1:
        # Implicit output is ellipsis (...) + labels seen only once
        perm_index = ell_num_dim
        found_ell = True
        for label, _label_count in enumerate(label_count):
            if _label_count == 1:
                label_perm_index[label] = perm_index
                perm_index += 1
    else:
        rhs = equation[arrow_pos + 2 :]
        ell_skip = 0
        for i, label in enumerate(rhs):
            if label == " ":
                continue
            if label == ".":
                if ell_skip != 0:
                    ell_skip -= 1
                    continue
                assert (
                    not found_ell
                ), "einsum(): found '.' for output but an ellipsis (...) was already found"
                assert (
                    i + 2 < len(rhs) and rhs[i + 1] == "."
                ), "einsum(): found '.' for output that is not part of any ellipsis (...)"
                ell_skip = 2
                ell_index = perm_index
                perm_index += ell_num_dim
                found_ell = True
            else:
                assert str.isalpha(label), (
                    f"einsum(): invalid subscript given at index {len(lhs) + 2 + i} "
                    f"in the equation string, subscripts must be in [a-zA-Z]"
                )

                index = einsum_label_to_index(label)
                label_perm_index[index] = perm_index
                perm_index += 1

    out_size = perm_index
    if not found_ell:
        ell_index = perm_index
        perm_index += ell_num_dim

    for label in range(TOTAL_LABELS):
        if label_count[label] > 0 and label_perm_index[label] == -1:
            label_perm_index[label] = perm_index
            perm_index += 1

    # Here we unsqueeze missing dimensions to make all operands have the same
    # number of dimensions. We take diagonals for repeated labels within the
    # same operand. Finally we permute the operands to align dimensions as
    # per the perm_out_index we computed above.
    permuted_operands = []
    for i, operand in enumerate(operands):
        perm_shape = [-1] * perm_index
        label_dim = [-1] * TOTAL_LABELS
        operand = operands[i]
        labels = op_labels[i]
        original_sizes = operand.shape

        j = 0
        for label in labels:
            if label == ELLIPSIS:
                # Add missing dimensions covered by the ellipsis
                num_missing_dim = ell_num_dim - (len(original_sizes) - len(labels) + 1)
                for k in range(num_missing_dim):
                    operand = torch.unsqueeze(operand, j)
                for k in range(ell_num_dim):
                    perm_shape[ell_index + k] = j
                    j += 1
            elif label_dim[label] != -1:
                dim = label_dim[label]
                operand = torch.diagonal(operand, offset=0, dim1=dim, dim2=j)
                operand = torch.swapaxes(operand, -1, dim)
            else:
                label_dim[label] = j
                perm_shape[label_perm_index[label]] = j
                j += 1

        # Add dimensions for missing labels
        for idx, index in enumerate(perm_shape):
            if index == -1:
                operand = torch.unsqueeze(operand, -1)
                perm_shape[idx] = j
                j += 1

        operand = torch.permute(operand, tuple(perm_shape))
        permuted_operands.append(operand)

    # Check if operands broadcast and keep track of last operand with
    # dimension size != 1 for optimizing reductions
    dim_last_op = [0] * perm_index
    has_zero_size_dim = False
    for dim in range(perm_index):
        broadcast_size = permuted_operands[0].shape[dim]
        for i in range(1, len(operands)):
            dim_size = permuted_operands[i].shape[dim]
            if broadcast_size != dim_size and broadcast_size != 1 and dim_size != 1:
                raise RuntimeError(
                    "einsum(): operands do not broadcast with remapped shapes [original->remapped]"
                )
            if dim_size != 1:
                broadcast_size = dim_size
                dim_last_op[dim] = i
        has_zero_size_dim = has_zero_size_dim or (broadcast_size == 0)

    # Compute result
    result = permuted_operands[0]
    if has_zero_size_dim:
        out_shape = [-1] * out_size
        for i in range(out_size):
            out_shape[i] = permuted_operands[dim_last_op[i]].shape[i]
        return torch.zeros(out_shape)

    # Sum out or squeeze dimensions that are size 1 for all later operands
    dim = out_size
    for i in range(dim, perm_index):
        if dim_last_op[i] == 0:
            if result.shape[dim] == 1:
                result = torch.squeeze(result, dim)
                dim -= 1
            else:
                result = torch.sum(result, dim)
                dim -= 1
        dim += 1

    for i in range(1, num_ops):
        operand = permuted_operands[i]
        sum_dims = []

        # Sum out or squeeze dimensions that are size 1 for all later operands
        dim = out_size
        for j in range(dim, perm_index):
            if dim_last_op[j] < i:
                operand = torch.squeeze(operand, dim)
                dim -= 1
            elif dim_last_op[j] == i:
                if result.shape[dim] == 1:
                    operand = torch.sum(operand, dim)
                    result = torch.squeeze(result, dim)
                    dim -= 1
                else:
                    sum_dims.append(dim)
            dim += 1
        if len(sum_dims) == 0:
            result = result.mul(operand)
        elif len(sum_dims) == len(result.shape):
            result = result.flatten().dot(operand.flatten())
        else:
            result = sumproduct_pair(result, operand, sum_dims, False)
    return result


# flatten
def flatten(input, start_dim=1, end_dim=-1):
    if input.device.type == 'cpu':
        if end_dim < 0:
            end_dim = input.ndim + end_dim
        new_shape = input.shape[:start_dim] + (-1,) + input.shape[end_dim + 1:]
        return input.reshape(new_shape)
    return execute('flatten_ext', input, start_dim, end_dim)


# flip
def flip(input, dims):
    return execute('reverse_v2', input, dims)


# fliplr


# flipud


# kron


# rot90


# gcd


# histc


# histogram


# histogramdd


# meshgrid
def meshgrid(*tensors, indexing=None):
    if indexing is None:
        indexing = 'ij'
    return execute('meshgrid', tensors, indexing)


# lcm


# logcumsumexp

# ravel


# renorm


# repeat_interleave
def repeat_interleave(input, repeats, dim=None):
    if isinstance(repeats, int):
        return execute('repeat_interleave_int', input, repeats, dim, None)
    return execute('repeat_interleave_tensor', input, repeats, dim, None)


# roll
def roll(input, shifts, dims=None):
    return execute('roll', input, shifts, dims)


# searchsorted
def searchsorted(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None
):
    dtype = torch.int32 if bool(out_int32) else torch.int64
    if (side == "left" and right is True):
        raise ValueError(f"For 'searchsorted', side and right can't be set to opposites,"
                         f"got side of left while right was True.")
    if side == "right":
        right = True
    return execute('search_sorted', sorted_sequence, values, sorter,
                   dtype_to_type_id('SearchSorted', 'dtype', dtype), right)

# tensordot

# trace

# tril
def tril(input, diagonal=0):
    return execute('tril_ext', input, diagonal)


# tril_indices

# triu
def triu(input, diagonal=0):
    return execute('triu', input, diagonal)

# triu_indices


# unflatten
def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return x.reshape(new_shape)

# vander


# view_as_real

# view_as_complex


# resolve_conj


# resolve_neg


def masked_fill(input, mask, value):
    return execute('masked_fill', input, mask, value)

def finfo(dtype):
    return np.finfo(mindspore.dtype_to_nptype(dtype))


def iinfo(dtype):
    return np.iinfo(mindspore.dtype_to_nptype(dtype))


def contains(self, key):
    r"""
    Args:
        self (object): The object instance on which the method is called.
        key (object): The key to be checked for containment in the object.

    Returns:
        None: This function returns None, indicating whether the key is contained in the object.

    Raises:
        None
    """
    eq_res = eq(self, key)
    res = any(eq_res)
    return bool(res)


def stop_gradient(input):
    return execute('stop_gradient', input)


def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    return indices, dimension


def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = torch.Tensor(_indices)
    output = torch.gather(input, _dimension, indices)
    output = torch.swapaxes(output, _dimension + 1, -1)
    return output

def _run_primitive_op(prim_op, *args):
    return execute('_run_primitive_op', prim_op, *args)

__all__ = [
    "bincount",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "cdist",
    "clone",
    "contains",
    "cumsum",
    "diag",
    "diagonal",
    "dim_list_to_bitset",
    "einsum",
    "einsum_label_to_index",
    "finfo",
    "flatten",
    "flip",
    "iinfo",
    "masked_fill",
    "maybe_wrap_dim",
    "meshgrid",
    "repeat_interleave",
    "roll",
    "searchsorted",
    "stop_gradient",
    "sumproduct_pair",
    "tril",
    "triu",
    "unflatten",
    "unfold",
    "_run_primitive_op",
]
