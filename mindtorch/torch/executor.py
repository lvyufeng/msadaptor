import mindspore
from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode, Tensor as MSTensor
import mindspore.common._stub_tensor
from mindspore.common.api import _pynative_executor
from mindspore.common._stub_tensor import _convert_python_data

import torch
from ._tensor import Tensor
from .dispatcher import dispatcher

def execute(func_name, *args, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    user_created = kwargs.pop('user_created', False)
    out, device = dispatcher.dispatch(func_name, *args, **kwargs)
    out_tensor = Tensor(out, device=device)
    if requires_grad:
        out_tensor._requires_grad = True
    if user_created:
        out_tensor._user_created = True
        out_tensor.attach_grad()

    return out_tensor

