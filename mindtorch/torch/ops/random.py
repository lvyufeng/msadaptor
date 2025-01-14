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
def normal_(mean=0.0, std=1.0, *, size=None, generator=None, out=None):
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access

    is_mean_tensor = isinstance(mean, torch.Tensor)
    is_std_tensor = isinstance(std, torch.Tensor)

    if is_mean_tensor and is_std_tensor:
        output = execute("normal_tensor_tensor", mean, std, seed, offset)
    if is_mean_tensor and not is_std_tensor:
        output = execute("normal_tensor_float", mean, std, seed, offset)
    if not is_mean_tensor and is_std_tensor:
        output = execute("normal_float_tensor", mean, std, seed, offset)
    output = execute("normal_float_float", mean, std, size, seed, offset)
    if out is None:
        return out
    out.data = output
    return out


# uniform_
def uniform_(input, *args, **kwargs):
    if len(args) == 1:
        from_ = args[0]
        to_ = None
    elif len(args) == 2:
        from_ = args[0]
        to_ = args[1]
    elif len(args) == 3:
        from_ = args[0]
        to_ = args[1]

    from_ = kwargs.get("from", 0) if from_ is None else from_
    # to_ = kwargs.get("to", 1)
    generator_ = kwargs.get("generator", None)
    if generator_ is None:
        generator_ = default_generator
    seed, offset = default_generator._step(generator_step_)
    if input.device.type == 'npu':
        execute("inplace_uniform", input, from_, to_, seed, offset)
    elif input.device.type == 'cpu':
        input.data = rand(input.shape, generator=generator_) * (to_ - from_) + from_
    return input


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
    "normal_",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "randint_like",
    "uniform_"
]
