# from .imports import *
import torch.Tensor
import torch.std
import numpy
import statistics as stats

def variance(obj):
    if not hasattr(obj, '__getitem__'):
        return 0    
    elif isinstance(obj, torch.Tensor):
        return torch.std(obj)
    elif isinstance(obj, numpy.ndarray):
        return numpy.std(obj)
    else:
        return stats.stdev(obj)

#mean absolute deviation
# def mad(obj):
#     if not hasattr(obj, '__getitem__'):
#         return 0    
#     elif isinstance(obj, torch.Tensor):
#         return (obj - obj.mean()).abs().mean()
#     # elif isinstance(obj, numpy.ndarray):
#     #     return numpy.std(obj)
#     else:
#         return abs((obj - (sum(obj)/len(obj))))

# conveniently defined like so:
#latex->
# $$\operatorname{E}\left[X Y\right] - \operatorname{E}\left[X\right] \operatorname{E}\left[Y\right]$$
def covariance(obj1, obj2):
    if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
        return (obj1*obj2).mean() - obj1.mean()*obj2.mean()
    return 0

def std(obj):
    if not hasattr(obj, '__getitem__'):
        return 0    
    elif isinstance(obj, torch.Tensor):
        return torch.std(obj)
    elif isinstance(obj, numpy.ndarray):
        return numpy.std(obj)
    else:
        return stats.stdev(obj)


#stastic helper functions-----------------------------------------
def accuracy(out, yb): #
     return (torch.argmax(out, dim=1)==yb).float().mean()

def normalize(x, m, s):
    return (x-m)/s

def prev_pow_2(x): #consider converting to generator
    """returns the previous power of 2 of X"""
    return 2**math.floor(math.log2(x))