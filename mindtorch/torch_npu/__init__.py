from mindspore.ops.auto_generate.gen_arg_handler import str_to_enum

import torch
from torch.executor import execute
from torch.nn.functional import rms_norm, fast_gelu, swiglu

from . import npu
from . import profiler

torch.cuda = torch.npu
torch.Tensor.cuda = torch.Tensor.npu
torch.nn.Module.cuda = torch.nn.Module.npu

old_move_to = torch.Tensor._move_to


def _move_to(self, device, non_blocking=False):
    if device.type == "cuda":
        device = torch.device("npu", device.index)
    return old_move_to(self, device, non_blocking)


torch.Tensor._move_to = _move_to


old_device_init = torch.device.__init__


def _init(self, type=None, index=None):
    old_device_init(self, type, index)
    if self.type == "cuda":
        self.type = "npu"


torch.device.__init__ = _init

old_init_process_group = torch.distributed.init_process_group


def init_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = "",
    pg_options=None,
    device_id=None,
):
    if backend == "nccl":
        backend = "hccl"
    return old_init_process_group(
        backend,
        init_method,
        timeout,
        world_size,
        rank,
        store,
        group_name,
        pg_options,
        device_id,
    )


torch.distributed.init_process_group = init_process_group


def npu_rms_norm(x, gamma, epsilon=1e-5):
    output = rms_norm(x, gamma, epsilon)
    return output


def npu_swiglu(x, dim=-1):
    return swiglu(x, dim)


def npu_fusion_attention(
    query,
    key,
    value,
    head_num,
    input_layout,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tockens=2147483647,
    next_tockens=2147483647,
    inner_precise=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
):

    output = execute(
        "flash_attention_score",
        query,
        key,
        value,
        pse,
        None,
        padding_mask,
        atten_mask,
        prefix,
        actual_seq_qlen,
        actual_seq_kvlen,
        head_num,
        keep_prob,
        scale,
        pre_tockens,
        next_tockens,
        inner_precise,
        str_to_enum('FlashAttentionScore', 'input_layout', input_layout),
        sparse_mode,
    )[3]
    return (output,)


def npu_apply_adam_w(
    beta1_power,
    beta2_power,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm,
    amsgrad,
    maximize,
    out,
):

    var, m, v = out
    var, m, v = execute(
        'apply_adamw',
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm,
        amsgrad,
        maximize,
    )
    return var, m, v


def npu_all_gather_base_mm(
    input_: torch.Tensor,
    x2: torch.Tensor,
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
    input_: torch.Tensor,
    x2: torch.Tensor,
    _: str,
    world_size: int,
    # reduce_op: str = ops.ReduceOp.SUM,
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
