from mindspore._c_expression import HookType
from mindspore._c_expression import Cell_, Tensor as MSTensor
from mindspore.common.api import _pynative_executor
from mindspore.ops import PrimitiveWithInfer

import torch

class HookBackward(PrimitiveWithInfer):
    def __init__(self):
        """Initialize HookBackward."""
        super(HookBackward, self).__init__(self.__class__.__name__)
        self.grad_output = None
        # self.set_hook_fn(hook_fn, HookType.HookBackward)

    def set_backward_hook(self, hook):
        def hook_backward_grad(grad):
            if self.grad_output is None:
                self.grad_output = grad
                # Indicates the first time of call backward hook, and need to wait for the second time call
                return None
            backward_hook_grad_input = grad
            res = hook(backward_hook_grad_input, self.grad_output)
            self.grad_output = None
            return res

        self.set_hook_fn(hook_backward_grad, HookType.HookBackward)

    def set_backward_prehook(self, hook):
        self.set_hook_fn(hook, HookType.HookBackward)

    def infer_shape(self, *inputs_shape):
        if len(inputs_shape) == 1:
            return inputs_shape[0]
        return inputs_shape

    def infer_dtype(self, *inputs_type):
        if len(inputs_type) == 1:
            return inputs_type[0]
        return inputs_type


class Node(Cell_):
    def __init__(self, name):
        super().__init__(name)
        self.backward_hook = None

    def construct(self, *args, **kwargs):
        raise NotImplementedError

    def simple_call(self, *args, **kwargs):
        if hasattr(self, 'bprop'):
            with torch.no_grad():
                output = self.construct(*args, **kwargs)
            _pynative_executor.call_custom_bprop(self, output, *args, **kwargs)
        else:
            output = self.construct(*args, **kwargs)
        return output


    def wrapped_call(self, *args, **kwargs):
        if self.backward_hook is not None:
            return self.call_with_hook(*args, **kwargs)
        return self.simple_call(*args, **kwargs)

    __call__ = wrapped_call

    def call_with_hook(self, *args, **kwargs):
        print('call_with_hook')
        args = self.backward_hook(*args)
        output = self.simple_call(*args, **kwargs)
        if isinstance(output, tuple):
            output = self.backward_hook(*output)
        else:
            output = self.backward_hook(output)
        return output

    def register_hook(self, hook):
        self.backward_hook = HookBackward()
        self.backward_hook.set_backward_hook(hook)
        self.__call__ = self.call_with_hook

    def register_prehook(self, hook):
        pass

class AccumulateGrad(Node):
    def __init__(self):
        super().__init__('AccumulateGrad')
        self.tensor = None

    def construct(self, input):
        self.tensor = input
        return input

    def bprop(self, input, output, grad):
        if self.tensor.grad is None:
            self.tensor.grad = grad
        else:
            self.tensor.grad += grad
        self.tensor = None
        return grad
