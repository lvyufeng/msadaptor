import uuid
import weakref
from copy import deepcopy
import numpy as np

import mindspore
from mindspore._c_expression import Tensor as MSTensor, ParamInfo
from mindspore._c_expression import TensorNode
from mindspore.ops import GradOperation
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import typing

import torch
from torch.dispatcher import dispatcher, device_map
from .types import device as device_

grad_ = GradOperation(False, True, True)

class Tensor:
    tensor = None
    stub = None
    _requires_grad = False
    grad = None
    fn = None
    is_leaf = False
    _retain_grad = False

    def __init__(self, *input, device=None, dtype=None): # pylint: disable=super-init-not-called
        if device is None:
            device = device_('cpu')
        self.device = device

        if isinstance(input[0], int):
            if dtype is None:
                dtype = mindspore.float32
            self.tensor = MSTensor(shape=input, dtype=dtype)
            self.is_leaf = True
        elif isinstance(input[0], TensorNode):
            self.stub = input[0]
        elif isinstance(input[0], MSTensor):
            self.tensor = input[0]
        elif isinstance(input[0], np.ndarray):
            self.tensor = MSTensor(input[0])
            self.is_leaf = True
        elif isinstance(input[0], Tensor):
            self.tensor = input[0].tensor
            self.stub = input[0].stub
            self.requires_grad_(input[0].requires_grad)
            self.is_leaf = input[0].is_leaf
        else:
            raise ValueError(f'not support data type {type(input[0])}')
        self.__class__ = dtype_class_map[self.dtype]

    @staticmethod
    def _make_subclass(cls, tensor, requires_grad=None):
        """
        Manually implement the functionality of torch.Tensor._make_subclass.

        Args:
            cls (type): The subclass type to create.
            tensor (torch.Tensor): The base tensor to wrap.
            requires_grad (bool, optional): Whether the subclassed tensor requires gradients.

        Returns:
            cls: An instance of the subclass wrapping the base tensor.
        """
        if not issubclass(cls, torch.Tensor):
            raise TypeError(f"{cls} must be a subclass of torch.Tensor")
        
        # Create an uninitialized instance of the subclass
        subclass_instance = object.__new__(cls)
        
        # Initialize the core structure of the tensor
        Tensor.__init__(subclass_instance, tensor, device=tensor.device)

        # Set requires_grad if specified; otherwise, inherit from the base tensor
        if requires_grad is not None:
            subclass_instance.requires_grad_(requires_grad)
        else:
            subclass_instance.requires_grad_(tensor.requires_grad)

        return subclass_instance

    @property
    def _data(self):
        if self.tensor is not None:
            return self.tensor
        return self.stub.get_value()

    @property
    def data(self):
        return Tensor(self._data)

    @data.setter
    def data(self, other):
        self.stub = None
        self.tensor = None
        if isinstance(other, Tensor):
            self.stub = other.stub
            self.tensor = other.tensor
        elif isinstance(other, TensorNode):
            self.stub = other
        elif isinstance(other, MSTensor):
            self.tensor = other
        else:
            raise ValueError(f'not support set type {type(other)} to Tensor.data')

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if not isinstance(self.dtype, typing.Float):
            raise RuntimeError('only Tensors of floating point and complex dtype can require gradients')
        self._requires_grad = requires_grad
        if self.tensor is not None and requires_grad:
            if self.tensor.param_info is None:
                self.tensor.param_info = ParamInfo()
                self.tensor.param_info.name = str(uuid.uuid4())
            self.tensor.param_info.requires_grad = requires_grad
        if requires_grad and self.is_leaf and not hasattr('self', 'attach_grad_hook'):
            self.attach_grad()
            self._retain_grad = True

    def attach_grad(self):
        weak_self = weakref.ref(self)
        def hook(grad):
            if weak_self()._retain_grad:
                if weak_self().grad is None:
                    weak_self().grad = Tensor(grad)
                else:
                    weak_self().grad += Tensor(grad)
            return grad
        self.attach_grad_hook = self.register_hook(hook)

    def retain_grad(self):
        if self.is_leaf:
            self._retain_grad = True
        else:
            self._retain_grad = True
            self.attach_grad()

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            if not hasattr(self, "stub_shape"):
                self.stub_shape = self.stub.get_shape()
            return self.stub_shape
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            if not hasattr(self, "stub_dtype"):
                self.stub_dtype = self.stub.get_dtype()
            return self.stub_dtype
        return self.tensor.dtype

    def cpu(self):
        return self.to(device_('cpu'))

    def npu(self, device=None, non_blocking=False):
        if device is None:
            device = device_('npu', 0)
        return self.to(device, non_blocking=non_blocking)

    def cuda(self, device=None, non_blocking=False):
        if device is None:
            device = device_('gpu', 0)
        return self.to(device, non_blocking=non_blocking)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad

    def __hash__(self):
        return hash(id(self))

    def __len__(self):
        if self.shape == ():
            return 1
        return self.shape[0]

    def __repr__(self) -> str:
        data = self.cpu()._data
        data.data_sync(True)
        return data.__repr__()

    def __format__(self, format_spec):
        return np.ndarray.__format__(self.numpy(), format_spec)

    def __getitem__(self, slices):
        return Tensor(self._data.__getitem__(slices))

    def __setitem__(self, key, value):
        """"""
        # return torch.ops.array.setitem(self, key, value)
        tensor_operator_registry.get("__setitem__")(self.data, key, value)

    def __add__(self, other):
        return torch.ops.add(self, other)

    def __iadd__(self, other):
        self.data = torch.ops.add(self, other)
        return self

    def __radd__(self, other):
        return Tensor.__add__(other, self)

    def __truediv__ (self, other):
        data, _ = dispatcher.dispatch('div', self, other)
        self.data = data
        return self

    def __rtruediv__ (self, other):
        return Tensor.__truediv__(other, self)

    def __ne__(self, other):
        data, _ = dispatcher.dispatch('not_equal', self, other)
        self.data = data
        return self

    def __neg__(self):
        data, _ = dispatcher.dispatch('neg', self)
        self.data = data
        return self

    def __mul__(self, other):
        data, _ = dispatcher.dispatch('mul', self, other)
        self.data = data
        return self

    def __rmul__(self, other):
        return Tensor.__mul__(other, self)

    def __pow__(self, other):
        data, _ = dispatcher.dispatch('pow', self, other)
        self.data = data
        return self

    def __sub__(self, other):
        data, _ = dispatcher.dispatch('sub', self, other)
        self.data = data
        return self

    def __rsub__(self, other):
        return Tensor.__sub__(other, self)

    def __eq__(self, other):
        data, _ = dispatcher.dispatch('equal', self, other)
        self.data = data
        return self

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        assert self.requires_grad, "called backward on non-requires-grad tensor"
        if gradient is None:
            if self.shape == ():
                gradient = tensor(1, dtype=self.dtype, device=self.device)
            else:
                raise RuntimeError("grad must specified for non-0-tensor")
        _pynative_executor.end_graph(self.fn, self.data)
        from torch.executor import weights_dict
        weights = weights_dict.pop(self.fn)
        _pynative_executor.check_run(grad_, self.fn, weights, None, gradient)
        _pynative_executor.grad(self.fn, grad_, weights, None, gradient)

    # Tensor.new_tensor
    def new_tensor(self, data, *, dtype=None):
        if dtype is not None:
            dtype = self.dtype
        return Tensor(data, dtype)

    # Tensor.new_full
    def new_full(self, size, fill_value, *, dtype=None):
        return torch.ops.full

    # Tensor.new_empty


    # Tensor.new_ones


    # Tensor.new_zeros


    # Tensor.ndim
    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    # Tensor.real


    # Tensor.imag

    # Tensor.nbytes


    # Tensor.itemsize


    # Tensor.abs


    # Tensor.abs_


    # Tensor.absolute


    # Tensor.absolute_


    # Tensor.acos


    # Tensor.acos_


    # Tensor.arccos


    # Tensor.arccos_


    # Tensor.add
    def add(self, other, *, alpha=1):
        return torch.ops.add(self, other, alpha=alpha)

    # Tensor.add_
    def add_(self, other, *, alpha=1):
        out = torch.ops.add(self, other, alpha=alpha)
        self.data = out

    # Tensor.addbmm


    # Tensor.addbmm_


    # Tensor.addcdiv

    # Tensor.addcdiv_


    # Tensor.addcmul


    # Tensor.addcmul_


    # Tensor.addmm


    # Tensor.addmm_


    # Tensor.sspaddmm


    # Tensor.addmv


    # Tensor.addmv_


    # Tensor.addr


    # Tensor.addr_


    # Tensor.adjoint


    # Tensor.allclose


    # Tensor.amax


    # Tensor.amin


    # Tensor.aminmax


    # Tensor.angle


    # Tensor.apply_


    # Tensor.argmax
    def argmax(self, dim=None, keepdim=False):
        out = torch.ops.argmax(self, dim, keepdim)
        return out

    # Tensor.argmin


    # Tensor.argsort


    # Tensor.argwhere


    # Tensor.asin


    # Tensor.asin_


    # Tensor.arcsin


    # Tensor.arcsin_


    # Tensor.as_strided


    # Tensor.atan


    # Tensor.atan_


    # Tensor.arctan


    # Tensor.arctan_


    # Tensor.atan2


    # Tensor.atan2_


    # Tensor.arctan2


    # Tensor.arctan2_


    # Tensor.all


    # Tensor.any

    # Tensor.baddbmm


    # Tensor.baddbmm_


    # Tensor.bernoulli


    # Tensor.bernoulli_


    # Tensor.bfloat16


    # Tensor.bincount


    # Tensor.bitwise_not


    # Tensor.bitwise_not_


    # Tensor.bitwise_and


    # Tensor.bitwise_and_


    # Tensor.bitwise_or


    # Tensor.bitwise_or_


    # Tensor.bitwise_xor


    # Tensor.bitwise_xor_


    # Tensor.bitwise_left_shift


    # Tensor.bitwise_left_shift_


    # Tensor.bitwise_right_shift


    # Tensor.bitwise_right_shift_


    # Tensor.bmm


    # Tensor.bool


    # Tensor.byte


    # Tensor.broadcast_to
    def broadcast_to(self, shape):
        raise NotImplementedError
        # return torch.ops.broadcast_to(self, shape)

    # Tensor.cauchy_


    # Tensor.ceil


    # Tensor.ceil_


    # Tensor.char


    # Tensor.cholesky


    # Tensor.cholesky_inverse


    # Tensor.cholesky_solve


    # Tensor.chunk
    def chunk(self, chunks, dim=0):
        return torch.ops.chunk(self, chunks, dim)

    # Tensor.clamp


    # Tensor.clamp_


    # Tensor.clip


    # Tensor.clip_


    # Tensor.clone
    def clone(self):
        return deepcopy(self)

    # Tensor.contiguous
    def contiguous(self):
        return self

    # Tensor.copy_
    def copy_(self, value):
        if isinstance(value, Tensor):
            self.stub = value.stub
            self.tensor = value.tensor
        elif isinstance(value, MSTensor):
            self.stub = None
            self.tensor = value
        else:
            raise ValueError(f'not support type: {type(value)}')

    # Tensor.conj


    # Tensor.conj_physical


    # Tensor.conj_physical_


    # Tensor.resolve_conj


    # Tensor.resolve_neg


    # Tensor.copysign


    # Tensor.copysign_


    # Tensor.cos


    # Tensor.cos_


    # Tensor.cosh


    # Tensor.cosh_


    # Tensor.corrcoef


    # Tensor.count_nonzero


    # Tensor.cov


    # Tensor.acosh


    # Tensor.acosh_


    # Tensor.arccosh


    # Tensor.arccosh_


    # Tensor.cpu


    # Tensor.cross


    # Tensor.cuda


    # Tensor.logcumsumexp


    # Tensor.cummax


    # Tensor.cummin


    # Tensor.cumprod


    # Tensor.cumprod_


    # Tensor.cumsum

    # Tensor.cumsum_


    # Tensor.chalf


    # Tensor.cfloat


    # Tensor.cdouble


    # Tensor.data_ptr


    # Tensor.deg2rad


    # Tensor.dequantize


    # Tensor.det


    # Tensor.dense_dim


    # Tensor.detach


    # Tensor.detach_


    # Tensor.diag


    # Tensor.diag_embed


    # Tensor.diagflat


    # Tensor.diagonal


    # Tensor.diagonal_scatter

    # Tensor.fill_diagonal_


    # Tensor.fmax


    # Tensor.fmin


    # Tensor.diff


    # Tensor.digamma


    # Tensor.digamma_


    # Tensor.dim


    # Tensor.dim_order


    # Tensor.dist


    # Tensor.div


    # Tensor.div_


    # Tensor.divide


    # Tensor.divide_


    # Tensor.dot


    # Tensor.double


    # Tensor.dsplit


    # Tensor.element_size


    # Tensor.eq
    def eq(self, other):
        return torch.ops.eq(self, other)

    # Tensor.eq_
    def eq_(self, other):
        out = torch.ops.eq(self, other)
        self.data = out

    # Tensor.equal
    def equal(self, other):
        return torch.ops.eq(self, other)

    # Tensor.erf


    # Tensor.erf_


    # Tensor.erfc


    # Tensor.erfc_


    # Tensor.erfinv


    # Tensor.erfinv_


    # Tensor.exp


    # Tensor.exp_


    # Tensor.expm1


    # Tensor.expm1_


    # Tensor.expand
    def expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    # Tensor.expand_as


    # Tensor.exponential_


    # Tensor.fix


    # Tensor.fix_


    # Tensor.fill_


    # Tensor.flatten
    def flatten(self, start_dim=0, end_dim=-1):
        return torch.ops.flatten(self, start_dim, end_dim)

    # Tensor.flip


    # Tensor.fliplr


    # Tensor.flipud


    # Tensor.float


    # Tensor.float_power


    # Tensor.float_power_


    # Tensor.floor


    # Tensor.floor_


    # Tensor.floor_divide


    # Tensor.floor_divide_


    # Tensor.fmod


    # Tensor.fmod_


    # Tensor.frac


    # Tensor.frac_


    # Tensor.frexp


    # Tensor.gather


    # Tensor.gcd


    # Tensor.gcd_


    # Tensor.ge


    # Tensor.ge_


    # Tensor.greater_equal


    # Tensor.greater_equal_


    # Tensor.geometric_


    # Tensor.geqrf


    # Tensor.ger


    # Tensor.get_device


    # Tensor.gt


    # Tensor.gt_


    # Tensor.greater


    # Tensor.greater_


    # Tensor.half


    # Tensor.hardshrink


    # Tensor.heaviside


    # Tensor.histc


    # Tensor.histogram


    # Tensor.hsplit


    # Tensor.hypot


    # Tensor.hypot_


    # Tensor.i0


    # Tensor.i0_


    # Tensor.igamma


    # Tensor.igamma_


    # Tensor.igammac


    # Tensor.igammac_


    # Tensor.index_add_


    # Tensor.index_add


    # Tensor.index_copy_


    # Tensor.index_copy


    # Tensor.index_fill_


    # Tensor.index_fill


    # Tensor.index_put_


    # Tensor.index_put


    # Tensor.index_reduce_


    # Tensor.index_reduce

    # Tensor.index_select


    # Tensor.indices


    # Tensor.inner


    # Tensor.int


    # Tensor.int_repr


    # Tensor.inverse


    # Tensor.isclose


    # Tensor.isfinite


    # Tensor.isinf


    # Tensor.isposinf


    # Tensor.isneginf


    # Tensor.isnan


    # Tensor.is_contiguous


    # Tensor.is_complex


    # Tensor.is_conj


    # Tensor.is_floating_point


    # Tensor.is_inference


    # Tensor.is_leaf


    # Tensor.is_pinned


    # Tensor.is_set_to


    # Tensor.is_shared


    # Tensor.is_signed


    # Tensor.is_sparse


    # Tensor.istft


    # Tensor.isreal


    # Tensor.item
    def item(self):
        return self.numpy().item()

    # Tensor.kthvalue


    # Tensor.lcm


    # Tensor.lcm_


    # Tensor.ldexp


    # Tensor.ldexp_


    # Tensor.le


    # Tensor.le_


    # Tensor.less_equal


    # Tensor.less_equal_


    # Tensor.lerp


    # Tensor.lerp_


    # Tensor.lgamma


    # Tensor.lgamma_


    # Tensor.log


    # Tensor.log_


    # Tensor.logdet


    # Tensor.log10


    # Tensor.log10_


    # Tensor.log1p


    # Tensor.log1p_


    # Tensor.log2


    # Tensor.log2_


    # Tensor.log_normal_


    # Tensor.logaddexp


    # Tensor.logaddexp2


    # Tensor.logsumexp


    # Tensor.logical_and


    # Tensor.logical_and_


    # Tensor.logical_not


    # Tensor.logical_not_


    # Tensor.logical_or


    # Tensor.logical_or_


    # Tensor.logical_xor


    # Tensor.logical_xor_


    # Tensor.logit


    # Tensor.logit_


    # Tensor.long


    # Tensor.lt


    # Tensor.lt_


    # Tensor.less


    # Tensor.less_


    # Tensor.lu


    # Tensor.lu_solve


    # Tensor.as_subclass


    # Tensor.map_


    # Tensor.masked_scatter_


    # Tensor.masked_scatter


    # Tensor.masked_fill_


    # Tensor.masked_fill


    # Tensor.masked_select


    # Tensor.matmul


    # Tensor.matrix_power


    # Tensor.matrix_exp


    # Tensor.max


    # Tensor.maximum


    # Tensor.mean
    def mean(self, dim=None, keepdim=False, *, dtype=None):
        return torch.ops.mean(self, dim, keepdim, dtype=dtype)

    # Tensor.module_load


    # Tensor.nanmean


    # Tensor.median


    # Tensor.nanmedian


    # Tensor.min


    # Tensor.minimum


    # Tensor.mm


    # Tensor.smm


    # Tensor.mode


    # Tensor.movedim


    # Tensor.moveaxis


    # Tensor.msort


    # Tensor.mul


    # Tensor.mul_


    # Tensor.multiply


    # Tensor.multiply_


    # Tensor.multinomial


    # Tensor.mv


    # Tensor.mvlgamma


    # Tensor.mvlgamma_


    # Tensor.nansum


    # Tensor.narrow


    # Tensor.narrow_copy


    # Tensor.ndimension


    # Tensor.nan_to_num


    # Tensor.nan_to_num_


    # Tensor.ne


    # Tensor.ne_


    # Tensor.not_equal


    # Tensor.not_equal_


    # Tensor.neg


    # Tensor.neg_


    # Tensor.negative


    # Tensor.negative_


    # Tensor.nelement


    # Tensor.nextafter


    # Tensor.nextafter_


    # Tensor.nonzero


    # Tensor.norm


    # Tensor.normal_


    # Tensor.numel


    # Tensor.numpy
    def numpy(self):
        return self._data.asnumpy()

    def mindspore(self):
        return mindspore.Tensor(self._data)

    # Tensor.orgqr


    # Tensor.ormqr


    # Tensor.outer


    # Tensor.permute
    def permute(self, *dims):
        return torch.ops.permute(self, dims)

    # Tensor.pin_memory


    # Tensor.pinverse


    # Tensor.polygamma


    # Tensor.polygamma_


    # Tensor.positive


    # Tensor.pow


    # Tensor.pow_


    # Tensor.prod


    # Tensor.put_


    # Tensor.qr


    # Tensor.qscheme


    # Tensor.quantile


    # Tensor.nanquantile


    # Tensor.q_scale


    # Tensor.q_zero_point


    # Tensor.q_per_channel_scales


    # Tensor.q_per_channel_zero_points


    # Tensor.q_per_channel_axis


    # Tensor.rad2deg


    # Tensor.random_


    # Tensor.ravel


    # Tensor.reciprocal


    # Tensor.reciprocal_


    # Tensor.record_stream


    # Tensor.register_hook
    def register_hook(self, hook):
        return self._data.register_hook(hook)

    # Tensor.register_post_accumulate_grad_hook


    # Tensor.remainder


    # Tensor.remainder_


    # Tensor.renorm


    # Tensor.renorm_


    # Tensor.repeat


    # Tensor.repeat_interleave


    # Tensor.reshape
    def reshape(self, *shape):
        return torch.ops.reshape(self, shape)

    # Tensor.reshape_as


    # Tensor.resize_


    # Tensor.resize_as_


    # Tensor.retain_grad


    # Tensor.retains_grad


    # Tensor.roll


    # Tensor.rot90


    # Tensor.round


    # Tensor.round_


    # Tensor.rsqrt


    # Tensor.rsqrt_


    # Tensor.scatter


    # Tensor.scatter_


    # Tensor.scatter_add_


    # Tensor.scatter_add


    # Tensor.scatter_reduce_


    # Tensor.scatter_reduce


    # Tensor.select


    # Tensor.select_scatter


    # Tensor.set_


    # Tensor.share_memory_


    # Tensor.short


    # Tensor.sigmoid


    # Tensor.sigmoid_


    # Tensor.sign


    # Tensor.sign_


    # Tensor.signbit


    # Tensor.sgn


    # Tensor.sgn_


    # Tensor.sin


    # Tensor.sin_


    # Tensor.sinc


    # Tensor.sinc_


    # Tensor.sinh


    # Tensor.sinh_


    # Tensor.asinh


    # Tensor.asinh_


    # Tensor.arcsinh


    # Tensor.arcsinh_


    # Tensor.shape


    # Tensor.size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    # Tensor.slogdet


    # Tensor.slice_scatter


    # Tensor.softmax


    # Tensor.sort


    # Tensor.split


    # Tensor.sparse_mask


    # Tensor.sparse_dim


    # Tensor.sqrt


    # Tensor.sqrt_


    # Tensor.square


    # Tensor.square_


    # Tensor.squeeze


    # Tensor.squeeze_


    # Tensor.std


    # Tensor.stft


    # Tensor.storage


    # Tensor.untyped_storage


    # Tensor.storage_offset


    # Tensor.storage_type


    # Tensor.stride


    # Tensor.sub


    # Tensor.sub_


    # Tensor.subtract


    # Tensor.subtract_


    # Tensor.sum
    def sum(self, dim=None, keepdim=False, dtype=None):
        return torch.ops.sum(self, dim, keepdim, dtype=dtype)

    # Tensor.sum_to_size


    # Tensor.svd


    # Tensor.swapaxes


    # Tensor.swapdims


    # Tensor.t


    # Tensor.t_


    # Tensor.tensor_split


    # Tensor.tile


    # Tensor.to

    def _move_to(self, device, non_blocking=False):
        if self.device == device:
            return self
        else:
            device_str = device_map[device.type]
            data = self._data.move_to(device_str, blocking=not non_blocking)
            out = Tensor(data, device=device)
            return out

    def to(self, *args, **kwargs):
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        out = self
        for arg in args:
            if isinstance(arg, device_):
                out = Tensor._move_to(out, arg, non_blocking)
            elif isinstance(arg, mindspore.common.dtype.Type):
                out = torch.ops.cast(out, arg)
            elif isinstance(arg, Tensor):
                out = Tensor._move_to(out, arg.device, non_blocking)
                out = torch.ops.cast(out, arg.dtype)
        return out

    # Tensor.take


    # Tensor.take_along_dim


    # Tensor.tan


    # Tensor.tan_


    # Tensor.tanh


    # Tensor.tanh_


    # Tensor.atanh


    # Tensor.atanh_


    # Tensor.arctanh


    # Tensor.arctanh_


    # Tensor.tolist


    # Tensor.topk


    # Tensor.to_dense


    # Tensor.to_sparse


    # Tensor.to_sparse_csr


    # Tensor.to_sparse_csc


    # Tensor.to_sparse_bsr


    # Tensor.to_sparse_bsc


    # Tensor.trace


    # Tensor.transpose
    def transpose(self, dim0, dim1):
        return torch.ops.transpose(self, dim0, dim1)

    # Tensor.transpose_


    # Tensor.triangular_solve


    # Tensor.tril


    # Tensor.tril_


    # Tensor.triu


    # Tensor.triu_


    # Tensor.true_divide


    # Tensor.true_divide_


    # Tensor.trunc


    # Tensor.trunc_


    # Tensor.type


    # Tensor.type_as


    # Tensor.unbind


    # Tensor.unflatten
    def unflatten(self, dim, sizes):
        return torch.ops.unflatten(self, dim, sizes)

    # Tensor.unfold
    def unfold(self, dimension, size, step):
        _indices, _dimension = torch.ops.utils._get_unfold_indices(self.shape, dimension, size, step)
        output = torch.ops.tf_gather(self, _indices, axis=_dimension)
        return torch.ops.transpose(output, _dimension + 1, -1)

    # Tensor.uniform_
    def uniform_(self, *args, **kwargs):
        if len(args) == 1:
            from_ = args[0]
        elif len(args) == 2:
            from_ = args[0]
            to_ = args[1]
        elif len(args) == 3:
            from_ = args[0]
            to_ = args[1]

        from_ = kwargs.get('from', 0)
        to_ = kwargs.get('to', 1)
        generator_ = kwargs.get('generator',)
        pass

    # Tensor.unique


    # Tensor.unique_consecutive


    # Tensor.unsqueeze


    # Tensor.unsqueeze_


    # Tensor.values


    # Tensor.var


    # Tensor.vdot


    # Tensor.view
    def view(self, *shape):
        return self.reshape(*shape)

    # Tensor.view_as
    def view_as(self, other):
        return self.reshape(*other.shape)

    # Tensor.vsplit


    # Tensor.where


    # Tensor.xlogy


    # Tensor.xlogy_


    # Tensor.zero_

    # Tensor.detach
    def detach(self):
        return torch.ops.detach(self)

def tensor(data, *, dtype=None, device=None, requires_grad=False):
    data = MSTensor(data, dtype=dtype)
    tensor = Tensor(data).to(device)
    tensor.requires_grad_(requires_grad)
    tensor.is_leaf = True
    return tensor

def is_tensor(x):
    return isinstance(x, Tensor)

class FloatTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float32)

class DoubleTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float64)

class HalfTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float16)

class BFloat16Tensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.bfloat16)

class IntTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.int32)

class LongTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.int64)

class BoolTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.bool_)

dtype_class_map = {
    mindspore.float32: FloatTensor,
    mindspore.float16: HalfTensor,
    mindspore.bfloat16: BFloat16Tensor,
    mindspore.float64: DoubleTensor,
    mindspore.int32: IntTensor,
    mindspore.int64: LongTensor,
    mindspore.bool_: BoolTensor
}

__all__ = ['tensor', 'is_tensor', 'Tensor']