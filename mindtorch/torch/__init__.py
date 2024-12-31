import os
os.environ["FIX_TORCH_ERROR"] = "0"

import mindspore as ms
import mindtorch
from mindtorch import *

__version__ = "2.5.0"

import sys
def load_mod(name):
    exec("import "+name)
    return eval(name)

autograd = sys.modules["torch.autograd"] = load_mod("mindtorch.autograd")

cuda = load_mod("mindtorch.cuda")
sys.modules["torch.cuda"] = load_mod("mindtorch.cuda")
sys.modules["torch.npu"] = load_mod("mindtorch.cuda")
npu = sys.modules["torch.npu"]
sys.modules["torch.cuda.amp"] = load_mod("mindtorch.cuda.amp")
sys.modules['torch.optim'] = load_mod("mindtorch.optim")
sys.modules['torch.optim.lr_scheduler'] = load_mod("mindtorch.optim")
mindtorch.optim.lr_scheduler = mindtorch.optim

sys.modules["torch.nn"] = load_mod("mindtorch.nn")
sys.modules["torch.nn.functional"] = load_mod("mindtorch.nn")
sys.modules["torch.nn.parallel"] = load_mod("mindtorch.distributed")
sys.modules["torch.nn.modules"] = load_mod("mindtorch.nn")
sys.modules['torch.nn.modules.module'] = load_mod("mindtorch.nn.modules.module")
sys.modules["torch.nn.parameter"] = load_mod("mindtorch.nn.parameter")
sys.modules["torch.nn.utils"] = load_mod("mindtorch.nn")
sys.modules["torch.utils"] = load_mod("mindtorch.utils")
sys.modules["torch._utils"] = load_mod("mindtorch._utils")
sys.modules["torch.utils.data"] = load_mod("mindtorch.utils.data")
sys.modules["torch.utils.data.sampler"] = load_mod("mindtorch.utils.data.sampler")
sys.modules["torch.utils.data.distributed"] = load_mod("mindtorch.utils.data")
sys.modules["torch.utils.checkpoint"] = load_mod("mindtorch.utils.checkpoint")
sys.modules["torch.utils.hooks"] = load_mod("mindtorch.utils.hooks")

sys.modules["torch.distributed"] = load_mod("mindtorch.distributed")
_C = sys.modules["torch._C"] = load_mod("mindtorch._C")
_six = sys.modules["torch._six"] = load_mod("mindtorch._six")
