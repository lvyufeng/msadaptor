from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode
from mindspore.common.api import _pynative_executor
from ._tensor import Tensor
from .dispatcher import dispatcher

fn_count = 0
weights_dict = {}

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
        return Tensor(val, device=device)
    if isinstance(stub, AnyTypeNode):
        val = stub.get_real_node()
        return _convert_stub(val, device)
    return Tensor(stub, device=device)

def create_function(func_name):
    # Dynamically create a function using exec
    exec(f"def {func_name}(): pass", globals())
    return globals()[func_name]

def execute(func_name, *args):
    global fn_count
    global weights_dict
    funcs = {arg.fn for arg in args if isinstance(arg, Tensor)}
    assert len(funcs) == 1
    func = funcs.pop()
    requires_grad = any([arg.requires_grad for arg in args if isinstance(arg, Tensor)])
    if requires_grad and func is None:
        func = create_function('top_cell_func_' + str(fn_count))
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(func)
        fn_count += 1
        weights_dict[func] = []
    
    params = [arg for arg in args if hasattr(arg, 'tensor') and arg.tensor.param_info is not None]
    weights_dict[func].extend(params)
    out, device = dispatcher.dispatch(func_name, *args)
    out_tensor = _convert_stub(out, device=device)
    out_tensor.requires_grad_(requires_grad)
    out_tensor.fn = func
    return out_tensor
