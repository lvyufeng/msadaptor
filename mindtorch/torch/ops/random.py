"""random op"""

from mindspore.common.generator import default_generator
from mindspore.ops.auto_generate.gen_arg_handler import dtype_to_type_id

import torch
from torch.executor import execute
from .._bind import get_default_dtype, get_default_device

generator_step_ = 12


# bernoulli
def bernoulli(input, *, generator=None, out=None):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    output = execute("bernoulli_ext", input, seed, offset)
    if out is None:
        return output
    out.data = output
    return out


# multinomial
def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    """custom multinomial"""
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    output = execute("multinomial_ext", input, num_samples, replacement, seed, offset)
    if out is None:
        return output
    out.data = output
    return out


# normal
def normal(mean=0.0, std=1.0, *, size=None, generator=None, out=None,
           dtype=None, layout=None, device=None, pin_memory=None, requires_grad=False):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if device is None:
        if out is None:
            device = get_default_device()
        else:
            device = out.device

    is_mean_tensor = isinstance(mean, torch.Tensor)
    is_std_tensor = isinstance(std, torch.Tensor)

    if device.type == 'cpu':
        if is_mean_tensor and is_std_tensor:
            size = (mean * std).shape
        if is_mean_tensor and not is_std_tensor:
            size = mean.shape
        if not is_mean_tensor and is_std_tensor:
            size = std.shape
        if out is not None:
            size = out.shape
        output = execute('normal', size)
        output = output * std - mean

    else:
        if is_mean_tensor and is_std_tensor:
            output = execute("normal_tensor_tensor", mean, std, seed, offset, device=device)
        if is_mean_tensor and not is_std_tensor:
            output = execute("normal_tensor_float", mean, std, seed, offset, device=device)
        if not is_mean_tensor and is_std_tensor:
            output = execute("normal_float_tensor", mean, std, seed, offset, device=device)
        if out is not None:
            size = out.shape
        output = execute("normal_float_float", float(mean), float(std), size, seed, offset, device=device)

    if out is None:
        return output
    out.data = output
    return out

# poisson


# rand
def rand(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_default_device()
    if isinstance(device, str):
        device = torch.device(device)
    if dtype is None:
        dtype = get_default_dtype()
    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    if device.type == 'cpu':
        output = execute('uniform_real', size,
                         device=device, requires_grad=requires_grad, user_created=True).to(dtype)
    else:
        output = execute(
            "rand_ext",
            size,
            seed,
            offset,
            dtype_to_type_id("RandExt", "dtype", dtype),
            device=device,
            requires_grad=requires_grad,
            user_created=True,
        )
    if out is None:
        return output
    out.data = output
    return out


# rand_like
def rand_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format
):
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "rand_like_ext",
        input,
        seed,
        offset,
        dtype_to_type_id("RandLikeExt", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
    )


# randint
def randint(
    *args,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    if dtype is None:
        dtype = torch.int64
    if device is None:
        device = get_default_device()
    if isinstance(device, str):
        device = torch.device(device)

    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    args = list(args)
    if len(args) == 2:
        args = [0] + args
    args.extend([seed, offset])
    output = execute(
        "randint_ext",
        *args,
        dtype_to_type_id("RandInt", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
        user_created=True
    )
    if out is None:
        return output
    out.data = output
    return out


# randint_like
def randint_like(
    input,
    low,
    high=0,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None
):
    if high == 0:
        low, high = 0, low
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "randint_like_ext",
        input,
        low,
        high,
        seed,
        offset,
        dtype_to_type_id("RandIntLike", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
    )


# randn
def randn(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_default_device()
    if isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = get_default_dtype()
    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    if size and isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute(
        "randn",
        size,
        seed,
        offset,
        dtype_to_type_id("Randn", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
        user_created=True,
    )
    if out is None:
        return output
    out.data = output
    return out


# randn_like
def randn_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None
):
    if device is None:
        device = input.device
    if isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = input.dtype
    seed, offset = default_generator._step(  # pylint: disable=protected-access
        generator_step_
    )
    return execute(
        "rand_like_ext",
        input,
        seed,
        offset,
        dtype_to_type_id("RandnLike", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
    )


# randperm
def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False
):
    if device is None:
        device = get_default_device()
    if isinstance(device, str):
        device = torch.device(device)

    if not generator:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    output = execute(
        "randperm_ext",
        n,
        seed,
        offset,
        dtype_to_type_id("RandpermExt", "dtype", dtype),
        device=device,
        requires_grad=requires_grad,
    )
    if out is None:
        return output
    out.data = output
    return out


__all__ = [
    "bernoulli",
    "multinomial",
    "normal",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "randint_like",
]
