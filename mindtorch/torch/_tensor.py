import copy
import mindspore
from mindspore import Tensor, ops
from mindspore.common._stub_tensor import StubTensor
from mindspore._c_expression import Tensor as Tensor_

from ._utils import _rebuild_tensor_v2
from ._C.size import Size
from .ops import transpose, mean, repeat_interleave, unsqueeze, pow

MS_PT_DTYPE_MAP = {
    'Float32': 'torch.cuda.FloatTensor',
    'BFloat16': 'torch.cuda.BFloat16Tensor',
    'Float16': 'torch.cuda.HalfTensor',
}

def type_(self, dtype=None):
    if dtype is None:
        return MS_PT_DTYPE_MAP[str(self.dtype)]
    return self.to(dtype)

Tensor.type = type_
StubTensor.type = type_

def retain_grad(self):
    def _tensor_hook(grad):
        self.grad = grad
    self.register_hook(_tensor_hook)

Tensor.retain_grad = retain_grad
StubTensor.retain_grad = retain_grad

@property
def shape(self):
    if isinstance(self, StubTensor):
        if self.stub is not None:
            stub_shape = self.stub.get_shape()
        else:
            stub_shape = self.tensor.shape
        return Size(stub_shape)
    return Size(self._shape)

Tensor.shape = shape
StubTensor.shape = shape

def to_dense(self):
    return self

Tensor.to_dense = to_dense
StubTensor.to_dense = to_dense

Tensor._base = None
StubTensor._base = None

@property
def data(self):
    return self

@data.setter
def data(self, new_value):
    if isinstance(self, StubTensor) and isinstance(new_value, StubTensor):
        self.stub = new_value.stub
    else:
        self.assign_value(new_value)

Tensor.data = data
StubTensor.data = data

def numel(self):
    return ops.size(self)

Tensor.numel = numel
setattr(StubTensor, 'numel', numel)
Tensor.nelement = numel
StubTensor.nelement = numel

StubTensor.__hash__ = Tensor.__hash__

def _repeat(self, *sizes):
    return ops.tile(self, tuple(sizes))

Tensor.repeat = _repeat
StubTensor.repeat = _repeat

def move_to_cuda(self, non_blocking=False):
    return Tensor(self.move_to("Ascend", not non_blocking))

def move_to_cpu(self, non_blocking=False):
    return Tensor(self.move_to("CPU", not non_blocking))

Tensor.cuda = move_to_cuda
StubTensor.cuda = move_to_cuda
Tensor.cpu = move_to_cpu
StubTensor.cpu = move_to_cpu


def size(self, dim=None):
    if dim is None:
        return self.shape
    assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
    return self.shape[dim]

Tensor.size = size
StubTensor.size = size

def dim(self):
    return self.ndim

Tensor.dim = dim
StubTensor.dim = dim

def clone(self):
    return copy.deepcopy(self)

Tensor.clone = clone
StubTensor.clone = clone

def __or__(self, other):
    if isinstance(other, (int, bool, float, Tensor)):
        return ops.bitwise_or(self.to(mindspore.int32), other.to(mindspore.int32)).bool()
    raise TypeError("Unsupported operand type(s) for |: 'Tensor' and '{}'".format(type(other)))

Tensor.__or__ = __or__
StubTensor.__or__ = __or__

Tensor.device = 'NO_INFO'
StubTensor.device = 'NO_INFO'

def div_(self, value, *, rounding_mode=None):
    out = self.div(value, rounding_mode=rounding_mode)
    self.assign_value(out)

Tensor.div_ = div_
StubTensor.div_ = div_

def __reduce_ex__(self, protocol):
    if isinstance(self, StubTensor):
        data = Tensor_(self.stub_sync())
    else:
        data = Tensor_(self)
    storage_offset = 0
    size = data._shape
    stride = data.stride()
    requires_grad = False
    args = (data, storage_offset, size, stride, requires_grad, None, None)
    return (
        _rebuild_from_type_v2, (_rebuild_tensor_v2, type(self), args, None))


Tensor.__reduce_ex__ = __reduce_ex__
StubTensor.__reduce_ex__ = __reduce_ex__

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret

def detach(self):
    return self

Tensor.detach = detach
StubTensor.detach = detach


Tensor.transpose = transpose
StubTensor.transpose = transpose


Tensor.mean = mean
StubTensor.mean = mean

Tensor.is_cuda = True
StubTensor.is_cuda = True

Tensor.repeat_interleave = repeat_interleave
StubTensor.repeat_interleave = repeat_interleave

def mul_(self, other):
    self.assign_value(self.mul(other))

Tensor.mul_ = mul_
StubTensor.mul_ = mul_

Tensor.is_sparse = False
StubTensor.is_sparse = False

def requires_grad_(self, requires_grad=True):
    self.requires_grad = requires_grad
    return self

Tensor.requires_grad_ = requires_grad_
StubTensor.requires_grad_ = requires_grad_

Tensor.unsqueeze = unsqueeze
StubTensor.unsqueeze = unsqueeze

def __pow__(self, exponent):
    return pow(self, exponent)

Tensor.__pow__ = __pow__
StubTensor.__pow__ = __pow__
