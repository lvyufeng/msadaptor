from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore._c_expression import pyboost_cast, pyboost_empty, pyboost_zeros, pyboost_ones
from mindspore.ops.operations.manually_defined.ops_def import Cast, Zeros, Ones
from mindspore.common.api import _pynative_executor
from mindspore.ops import FillV2

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

def empty_cpu(size, dtype):
    return pyboost_empty([size, dtype, 'CPU'])

__all__.append('empty_cpu')

zeros_op = Zeros().set_device('CPU')
def zeros_cpu(*args):
    return pyboost_zeros(zeros_op, args)

__all__.append('zeros_cpu')

ones_op = Ones().set_device('CPU')
def ones_cpu(*args):
    return pyboost_ones(ones_op, args)

__all__.append('ones_cpu')


squeeze_op = Squeeze().set_device('CPU')
def squeeze_cpu(*args):
    return pyboost_squeeze(squeeze_op, args)

__all__.append('squeeze_cpu')

stack_ext_op = StackExt().set_device('CPU')
def stack_ext_cpu(*args):
    return pyboost_stack_ext(stack_ext_op, args)

__all__.append('stack_ext_cpu')

tile_op = Primitive('Tile').set_device('CPU')
def tile_cpu(*args):
    return pyboost_tile(tile_op, args)

__all__.append('tile_cpu')

greater_equal_op = GreaterEqual().set_device('CPU')
def greater_equal_cpu(*args):
    return pyboost_greater_equal(greater_equal_op, args)

__all__.append('greater_equal_cpu')

isclose_op = IsClose().set_device('CPU')
def isclose_cpu(*args):
    return pyboost_isclose(isclose_op, args)

__all__.append('isclose_cpu')

range_op = Range().set_device('CPU')
def range_cpu(*args):
    return _pynative_executor.run_op_async(range_op, range_op.name, args)

__all__.append('range_cpu')

linspace_op = LinSpace().set_device('CPU')
def linspace_cpu(*args):
    return _pynative_executor.run_op_async(linspace_op, linspace_op.name, args)

__all__.append('linspace_cpu')

full_op = FillV2().set_device('CPU')
def full_cpu(*args):
    return _pynative_executor.run_op_async(full_op, full_op.name, args)

__all__.append('full_cpu')
