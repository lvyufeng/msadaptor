from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore._c_expression import pyboost_cast
from mindspore.ops.operations.manually_defined.ops_def import Cast

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
pyboost_op_list = [op.replace('pyboost_', '') + '_op' for op in pyboost_list]
aclop_list = list(filter(lambda s: s.endswith("_op") and not s in pyboost_op_list, dir(gen_ops_prim)))


pyboost_func = '''
def {name}(*args):
    return {pyboost}({op}, args)
'''

aclop_func = '''
def {name}(*args):
    return _pynative_executor.run_op_async({obj}, {obj}.name, args)
'''

__all__ = []

for op_name in pyboost_list:
    op = getattr(gen_ops_prim, op_name)
    func_name = op_name.replace('pyboost_', '') + '_cpu'
    prim_op = func_name.replace('_cpu', '_op')
    if not hasattr(gen_ops_prim, prim_op):
        continue
    __all__.append(func_name)
    globals()[prim_op] = getattr(gen_ops_prim, prim_op).__class__().set_device('CPU')
    exec(pyboost_func.format(name=func_name, pyboost=op_name, op=prim_op), globals())


for op_name in aclop_list:
    func_name = op_name.replace('_op', '_cpu')
    __all__.append(func_name)
    prim_op = func_name + '_prim'
    globals()[prim_op] = getattr(gen_ops_prim, op_name).__class__().set_device('CPU')
    exec(aclop_func.format(name=func_name, obj=prim_op), globals())

cast_op = Cast().set_device('CPU')
def cast_cpu(*args):
    return pyboost_cast(cast_op, args)

__all__.append('cast_cpu')
