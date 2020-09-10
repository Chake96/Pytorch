from .imports import *
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