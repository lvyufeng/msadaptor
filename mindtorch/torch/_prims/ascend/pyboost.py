from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore._c_expression import pyboost_cast
from mindspore.ops.operations.manually_defined.ops_def import Cast

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))


pyboost_func = '''
def {name}(*args):
    return {pyboost}({op}, args)
'''

__all__ = []

for op_name in pyboost_list:
    op = getattr(gen_ops_prim, op_name)
    func_name = op_name.replace('pyboost_', '') + '_npu'
    prim_op = func_name.replace('_npu', '_op')
    if not hasattr(gen_ops_prim, prim_op):
        continue
    __all__.append(func_name)
    globals()[prim_op] = getattr(gen_ops_prim, prim_op).__class__().set_device('Ascend')
    exec(pyboost_func.format(name=func_name, pyboost=op_name, op=prim_op), globals())

cast_op = Cast().set_device('Ascend')
def cast_npu(*args):
    return pyboost_cast(cast_op, args)

__all__.append('cast_npu')