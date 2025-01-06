from mindspore.ops import Primitive
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))

pyboost_func = '''
def {name}(*args):
    return {pyboost}({op}, args)
'''

for op_name in pyboost_list:
    op = getattr(gen_ops_prim, op_name)
    func_name = op_name.replace('pyboost_', '')
    prim_name = ''.join(list(map(str.capitalize, func_name.split('_'))))
    prim_op = func_name + '_prim'
    globals()[prim_op] = Primitive(prim_name).to_device('Ascend')
    exec(pyboost_func.format(name=func_name, pyboost=op_name, op=prim_op), globals())
