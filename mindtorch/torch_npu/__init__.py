
import torch
torch.cuda = torch.npu
torch.Tensor.cuda = torch.Tensor.npu

old_move_to = torch.Tensor._move_to
def _move_to(self, device, non_blocking=False):
    if device.type == 'cuda':
        device = torch.device('npu', device.index)
    return old_move_to(self, device, non_blocking)

torch.Tensor._move_to = _move_to


class device():
    def __init__(self, type=None, index=None):
        if type is not None:
            if isinstance(type, str):
                if ':' in type:
                    if index is not None:
                        raise ValueError("`type` must not include an index because index was "
                                         f"passed explicitly: {type}")
                    _target, _id = type.split(':')
                    _id = int(_id)
                else:
                    _target = type
                    _id = None if _target == 'cpu' else 0
            elif isinstance(type, device):
                if index is not None:
                    raise ValueError("torch.device(): When input is torch.device, `index` can not be set.")
                _target = type.type
                _id = type.index
            else:
                raise TypeError("torch.device(): `type` must be type of 'str' or 'torch.device'.")
        else:
            raise ValueError("torch.device(): `type` can not be None")

        if _target == 'cuda':
            _target = 'npu'
        self.type = _target
        self.index = _id

    def __repr__(self):
        if self.index is None:
            return f"device(type={self.type})"
        return f"device(type={self.type}, index={self.index})"

    def __eq__(self, __value):
        if not isinstance(__value, device):
            return False
        return hash(self) == hash(__value)

    def __hash__(self):
        return hash(str(self))

torch.device = device

from . import npu
from . import profiler
from torch.nn.functional import rms_norm, fast_gelu, swiglu
import mindspore as ms
from mindspore import ops
from mindspore.ops import auto_generate as gen

def npu_rms_norm(x, gamma, epsilon=1e-5):
    output = rms_norm(x, gamma, epsilon)
    return output

def npu_swiglu(x, dim=-1):
    return swiglu(x, dim)

def npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
                         prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                         gen_mask_parallel=True, sync=False):
    output = ops.flash_attention_score(query, key, value, real_shift=pse, padding_mask=padding_mask,
                                       attn_mask=atten_mask, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
                                       actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=keep_prob,
                                       scalar_value=scale, pre_tokens=pre_tockens, next_tokens=next_tockens,
                                       inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode)

    return (output,)

adamw_opt = gen.ApplyAdamW()

def npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2,
                     epsilon, grad, max_grad_norm, amsgrad, maximize, out):

    var, m, v = out
    var, m, v = adamw_opt(var, m, v, beta1_power=beta1_power, beta2_power=beta2_power, lr=lr, weight_decay=weight_decay,
                          beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad, max_grad_norm=max_grad_norm,
                          amsgrad=amsgrad, maximize=maximize)
    return var, m, v


def npu_all_gather_base_mm(
        input_: ms.Tensor,
        x2: ms.Tensor,
        _: str,
        world_size: int,
        bias: None = None,
        gather_index: int = 0,
        gather_output: bool = True,
        comm_turn: int = 0,
    ) -> None:
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    return ms.ops.all_gather_matmul(
        input_,
        x2,
        group,
        world_size,
        bias=bias,
        gather_index=gather_index,
        gather_output=gather_output,
        comm_turn=comm_turn,
    )


def npu_mm_reduce_scatter_base(
        input_: ms.Tensor,
        x2: ms.Tensor,
        _: str,
        world_size: int,
        reduce_op: str = ops.ReduceOp.SUM,
        bias: None = None,
        comm_turn: int = 0,
    ) -> None:
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    return ms.ops.matmul_reduce_scatter(
        input_,
        x2,
        group,
        world_size,
        reduce_op=reduce_op,
        bias=bias,
        comm_turn=comm_turn,
    )
