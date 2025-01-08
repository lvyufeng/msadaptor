from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import pyboost_empty

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
pyboost_op_list = [op.replace('pyboost_', '') + '_op' for op in pyboost_list]
aclop_list = list(filter(lambda s: s.endswith("_op") and not s in pyboost_op_list, dir(gen_ops_prim)))

aclop_func = '''
def {name}(*args):
    return _pynative_executor.run_op_async({obj}, {obj}.name, args)
'''

__all__ = []

for op_name in aclop_list:
    func_name = op_name.replace('_op', '_npu')
    __all__.append(func_name)
    prim_op = func_name + '_prim'
    globals()[prim_op] = getattr(gen_ops_prim, op_name).__class__().set_device('Ascend')
    exec(aclop_func.format(name=func_name, obj=prim_op), globals())

def empty_npu(size, dtype, device):
    return pyboost_empty([size, dtype, device])
