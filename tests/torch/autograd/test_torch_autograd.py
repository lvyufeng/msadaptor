import torch

def test_torch_autograd():
    x = torch.tensor(1.)
    x.requires_grad = True
    y = x + 1
    print(y)
    print(y.device)
    y.backward()
    print(x.grad)

def test_torch_autograd_retain_grad():
    x = torch.tensor(1.)
    y = torch.tensor(2.)
    print(y.tensor.param_info)
    x.requires_grad = True
    z = x + y
    zz = z + 1
    print(z.grad)
    z.retain_grad()
    zz.backward()
    print(z.grad)
    print(x.grad)
    print(y.grad)
