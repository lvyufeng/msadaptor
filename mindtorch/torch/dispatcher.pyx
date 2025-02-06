import torch
from torch.types import device as device_
from torch._prims import ascend, cpu
from collections import defaultdict
cimport cython

device_map = {
    'cpu': 'CPU',
    'npu': 'Ascend',
    'cuda': 'GPU'
}

# Singleton pattern with Cython optimization
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Dispatcher(metaclass=SingletonMeta):
    def __init__(self):
        # Using defaultdict to simplify code (no need to check key existence)
        self._registry = defaultdict(dict)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def register(self, func_name: str, device: str, func):
        # Register function in appropriate device dictionary
        self._registry[device][func_name] = func

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def dispatch(self, func_name: str, *args, **kwargs):
        device = kwargs.pop('device', None)

        if isinstance(device, str):
            device = device_(device)

        if device is None:
            tensors = [arg for arg in args[0] if torch.is_tensor(arg)] if isinstance(args[0], (tuple, list)) else [arg for arg in args if torch.is_tensor(arg)]
            
            if len(tensors) == 1:
                device = tensors[0].device
            else:
                devices = {tensor.device for tensor in tensors}
                if len(devices) > 1:
                    raise ValueError("All tensor arguments must be on the same device.")
                device = next(iter(devices), device_('cpu'))

        # Using `get` to avoid KeyError
        func = self._registry[device.type].get(func_name, None)
        if func is None:
            raise RuntimeError(f"No implementation for function: {func_name} on {device.type}.")
        
        return func(*args), device

# Singleton instance of Dispatcher
dispatcher = Dispatcher()

# Register functions for ascend and cpu
for func_name in ascend.__all__:
    dispatcher.register(func_name.replace('_npu', ''), 'npu', getattr(ascend, func_name))

for func_name in cpu.__all__:
    dispatcher.register(func_name.replace('_cpu', ''), 'cpu', getattr(cpu, func_name))
