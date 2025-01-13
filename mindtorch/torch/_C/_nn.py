import torch
from torch.types import device as device_

def _parse_to(*args, **kwargs):
    """
    Mimic torch._C._nn._parse_to functionality in Python.
    
    Args:
        tensor (torch.Tensor): The tensor to parse.
        *args: Positional arguments for `to`.
        **kwargs: Keyword arguments for `to`.

    Returns:
        torch.Tensor: The tensor with the desired properties.
    """
    if len(args) == 1:
        # Handle `device` or `dtype`
        if isinstance(args[0], torch.dtype):  # dtype only
            dtype = args[0]
            device = None
        elif isinstance(args[0], torch.device):  # device only
            device = args[0]
            dtype = None
        elif isinstance(args[0], str):
            device = device_(args[0])
            dtype = None
        else:
            raise TypeError(f"Expected torch.dtype or torch.device, but got {type(args[0])}")
    elif len(args) == 2:
        # Handle `device` and `dtype`
        dtype = args[1]
        device = args[0]
    else:
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)

    non_blocking = kwargs.get("non_blocking", False)
    memory_format = kwargs.get("memory_format", None)

    return device, dtype, non_blocking, memory_format
