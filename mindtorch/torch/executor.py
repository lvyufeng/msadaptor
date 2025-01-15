from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode
from mindspore.common.api import _pynative_executor
from mindspore.common._stub_tensor import _convert_python_data

import torch
from ._tensor import Tensor
from .dispatcher import dispatcher

def _convert_stub(stub, device):
    "convert stub to StubNode or Value"
    if isinstance(stub, TensorNode):
        return Tensor(stub, device=device)
    if isinstance(stub, tuple):
        return tuple(_convert_stub(e, device) for e in stub)
    if isinstance(stub, SequenceNode):
        elements = stub.get_elements()
        return tuple(_convert_stub(e, device) for e in elements)
    if isinstance(stub, NoneTypeNode):
        val = stub.get_real_value()
        return _convert_python_data(val)
    if isinstance(stub, AnyTypeNode):
        val = stub.get_real_node()
        return _convert_stub(val, device)
    return _convert_python_data(stub)


def execute(func_name, *args, **kwargs):
    requires_grad = kwargs.pop('requires_grad', None)
    user_created = kwargs.pop('user_created', False)
    if requires_grad is None:
        if isinstance(args[0], (tuple, list)):
            requires_grad = any([arg.requires_grad for arg in args[0] if torch.is_tensor(arg)])
        else:
            requires_grad = any([arg.requires_grad for arg in args if torch.is_tensor(arg)])

    out, device = dispatcher.dispatch(func_name, *args, **kwargs)
    out_tensor = _convert_stub(out, device=device)
    out_list = out_tensor if isinstance(out_tensor, tuple) else [out_tensor]
    for tensor in out_list:
        if torch.is_tensor(tensor):
            tensor._requires_grad = requires_grad
            tensor._user_created = user_created
    return out_tensor

