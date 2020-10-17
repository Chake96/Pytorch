from Data.helpers import *
from functools import partial
from Data.Dataset import ListContainer

def lsuv_module(model, module, x_mb):
    error_ceiling = 1e-3
    hook = ForwardHook(module, append_stat)
    while model(x_mb) is not None and abs(hook.mean) > error_ceiling: #correct the means
        module.bias -= hook.mean
    while model(x_mb) is not None and abs(hook.std - 1) > error_ceiling: #correct the standard deviations
        module.weight.data /= hook.std
    hook.remove()
    return hook.mean, hook.std

class Hooks(ListContainer):
    def __init__(self, ms, f): super().__init__([ForwardHook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
        
    def remove(self): #removes all registered hooks
        for h in self: h.remove()

class ForwardHook():
    def __init__(self, m, f): 
        self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()
