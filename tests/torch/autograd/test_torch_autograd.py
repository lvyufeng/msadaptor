import torch

def test_torch_autograd():
    x = torch.tensor(1.)
    x.requires_grad = True
    y = x + 1
    print(y)
    print(y.device)
    y.backward()
    print(x.grad)
