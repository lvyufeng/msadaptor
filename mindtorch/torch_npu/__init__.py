from . import npu
from . import profiler
from torch.nn.functional import rms_norm, fast_gelu, swiglu
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
