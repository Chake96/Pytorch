from fastai import datasets
from pathlib import Path
from IPython.core.debugger import set_trace
import os,PIL, mimetypes, pickle, gzip, math, torch, re, matplotlib as mpl, matplotlib.pyplot as plt
from functools import partial
import numpy
from typing import Iterable, Any
from torch import tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset,  SequentialSampler, RandomSampler
import torch.nn.functional as F
import torch.nn.init
from collections import OrderedDict
import random
from utilities.augmentations import *
from utilities.statistics import *
from utilities.2D_CNN import *

#os and path functions
def get_file_paths(path, files, extensions = None):
    path = Path(path) #can pass path as string or LibPath
    res = [path/f for f in files if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_file_names(path):
    return [file.name for file in os.scandir(path)]

def get_all_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = convert_to_set(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
            res += get_file_paths(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return get_file_paths(path, f, extensions)
#specific dataset functions
#MNIST



############helper functions
_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def convert_to_list(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def convert_to_set(obj):
    return obj if isinstance(obj, set) else set(convert_to_list(obj))

# def accuracy(out, yb): #
#      return (torch.argmax(out, dim=1)==yb).float().mean()

# def normalize(x, m, s):
#     return (x-m)/s

#normalize datasets
def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

def normalize_channels(x, mean, std):
    return (x-mean[...,None,None]) / std[...,None,None]

#normalizing imagenette dataset
_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
normalize_imagenette = partial(normalize_channels, mean=_m.cuda(), std=_s.cuda())

#image dataset helper functions
#show a PIL image, pass in Pytorch Tensor
def show_image(im, fig_size=(3,3)):
    if not isinstance(im, torch.Tensor):
        im = to_byte_tensor(im)
    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.imshow(im.permute(1,2,0))

#helper function for show_batch
def show_image_plot(im, ax=None, figsize=(3,3)):
    if ax is None: _,ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')
    ax.imshow(im.permute(1,2,0))

def show_batch(batch, num_columns=4, num_rows=None, fig_size=None):
    """plot a batch of images using matplotlib"""
    n = len(batch)
    if num_rows is None:
        num_rows = int(math.ceil(n/num_columns)) #make it squarish
    if fig_size is None:
        fig_size=(num_columns*3, num_rows*3)
    fig,axes = plt.subplots(num_rows, num_columns, figsize=(fig_size))
    for xi, ax in zip(batch, axes.flat):
        show_image_plot(xi, ax)

def mnist_resize(x):
    """resizes an image to 28x28 dimensions"""
    return x.view(-1, 1, 28, 28)

# #flatten an image to 
# def flatten(x):
#     return x.view(x.shape[0], -1) #removes 1,1 axis from result of AvgPool layer

#re-view an x variable at a specific size
def view_tfm(*size):
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

# def list_image_ext():
#     return set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

#statistics helper functions
#updated for Leaky Relu 
# def append_stats(hook, mod, inp, outp):
#     """collect statistics using Pytorch Hooks"""
#     if not hasattr(hook,'stats'): hook.stats = ([],[],[])
#     means,stds,hists = hook.stats
#     means.append(outp.data.mean().cpu())
#     stds .append(outp.data.std().cpu())
#     hists.append(outp.data.cpu().histc(40,-7,7))

# #hook for mean and std
# def append_mean_std(hook, model, inp, outp):
#     if not hasattr(hook, 'stats'):
#         hook.stats=([],[])
#     means, stds = hook.stats
#     if model.training:
#         means.append(outp.data.mean())
#         stds.append(outp.data.std())



# #creating 2d convolution models
# #passing in GeneralReLU args
# def get_cnn_layers(num_categories, num_features, layer, **kwargs):
#     num_features = [1] + num_features
#     return [layer(num_features[i], num_features[i+1], 5 if i==0 else 3, **kwargs)
#             for i in range(len(num_features)-1)] + [
#         nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(num_features[-1], num_categories)]

# #dont need bias, is using batchnorm
# def conv_layer(ni, num_features, ks=3, stride=2, batch_norm=True, **kwargs):
#     layers = [nn.Conv2d(ni, num_features, ks, padding=ks//2, stride=stride, bias = not batch_norm), 
#             GeneralReLU(**kwargs)]
#     if batch_norm:
#         layers.append(torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1))
#         # layers.append(Batch_Normalization(num_features))
#     return nn.Sequential(*layers)

# #0s all the bias weights, calls the passed initalizations function on each layer in the model
# def init_cnn_(model, func):
#     if isinstance(model, nn.Conv2d):
#         func(model.weight, a =0.1)
#         if getattr(model, 'bias', None) is not None:
#             model.bias.data.zero_()
#     for layer in model.children():
#         init_cnn_(layer, func)

# def init_cnn(model, uniform=False):
#     f = torch.nn.init.kaiming_uniform_ if uniform else torch.nn.init.kaiming_normal_
#     init_cnn_(model, f)
            
# def get_cnn_model(num_categories, num_features, layer, **kwargs):
#     return nn.Sequential(*get_cnn_layers(num_categories, num_features, layer, **kwargs))

# #returns the model and optimizer, will be refactored to be more versatile, currently just used for simple testing
# def get_model(training_dl, lr=0.5, nh=50):
#     m = training_dl.dataset.x_dataset.shape[1]
#     categories = training_dl.dataset.y_dataset.max().item()+1
#     model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,categories))
#     return model, optim.SGD(model.parameters(), lr=lr)

##updated CNN model maker


# def get_cnn_layers(train_dl, valid_dl, num_ch, num_cat, nfs,layer, **kwargs):
#     def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)
#     l1 = num_ch
#     l2 = prev_pow_2(l1*3*3)
#     #3x3 kernel sizes
#     layers =  [f(l1  , l2  , stride=1),
#                f(l2  , l2*2, stride=2),
#                f(l2*2, l2*4, stride=2)]
#     nfs = [l2*4] + nfs
#     layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]  #build the layers with proper input/output sizes
#     layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten),  #a typical last layer, has num_categories channels out
#                nn.Linear(nfs[-1], num_cat)]
#     return layers

def get_cnn_model(train_dl, valid_dl, num_chs, num_cat, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(train_dl, valid_dl, num_chs, num_cat, nfs, layer, **kwargs))

def get_model_runner(nfs, num_ch, num_cat, train_dl, valid_dl, in_layer, cbs_in=None, **kwargs):
    model = get_cnn_model(train_dl, valid_dl, num_ch, num_cat, nfs, in_layer, **kwargs)
    init_cnn(model)
#     return get_runner(model, data, lr=lr, cbs=cbs_in, opt_func=opt_func)
    return model, Runner(cb_funcs=cbs_in)

def get_data(path_in, encoding_in='latin-1'):
    path = datasets.download_data(path_in, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding=encoding_in)
    return map(tensor, (x_train,y_train,x_valid,y_valid))

# #takes a batch from a dataloader and applies the supplied runner callbacks to it
# def get_one_batch(dl, runner):
#     runner.xb, runner.yb = next(iter(dl))
#     for cb in runner.cbs:
#         cb.set_runner(runner)
#     runner('begin_batch')
#     return runner.xb, runner.yb



# #splits by grandparent directory using the passed in parent directory names
# def grandparent_splitter(fn, valid_name='valid', train_name='train'):
#     gp = fn.parent.parent.name
#     return True if gp==valid_name else False if gp==train_name else None

# #splits a list of items by a given function
# def split_by_function(items, func):
#     mask = [func(o) for o in items]
#     # `None` values will be filtered out
#     f_itms = [o for o,m in zip(items,mask) if m==False]
#     t_itms = [o for o,m in zip(items,mask) if m==True ]
#     return f_itms,t_itms

class Dataset():
    def __init__(self, x_ds, y_ds):
        self.x_dataset = x_ds
        self.y_dataset = y_ds
    
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, i):
        return self.x_dataset[i],self.y_dataset[i]


class SplitData():
    def __init__(self, train, valid): 
        self.train,self.valid = train,valid
    
    #trys to get the attribute from the training set
    def __getattr__(self,k):
        return getattr(self.train,k)

    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self,data:Any): 
        self.__dict__.update(data) 
    
    @classmethod
    def split_by_function(cls, itm_lst, func):
        lists = map(itm_lst.new, split_by_function(itm_lst.items, func)) #returns item lists of the same type it was given, uses New ctor
        return cls(*lists)

    def __repr__(self): 
        return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'

class LabeledData():
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)
        self.proc_x,self.proc_y = proc_x,proc_y
        
    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)
    
    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)
    
    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)

#composes a list of functions
#order_key -> ordering of applying functions
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(convert_to_list(funcs), key=key): x = f(x, **kwargs)
    return x


###Labelling Helper Functions
# def parent_labeler(fn): return fn.parent.name

# def label_by_func(sd, f, proc_x=None, proc_y=None):
#     train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
#     valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
#     return SplitData(train,valid)

def get_unique_keys(lst, sort=False):
    result = list(OrderedDict.fromkeys(lst).keys()) #get the keys
    if sort is True: result.sort()
    return result

class Processor(): 
    def process(self, items): return items

class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None
    
    def __call__(self, items):
        #The vocab is defined for the training set, so create the vocab
        if self.vocab is None:
            self.vocab = get_unique_keys(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)} #reverse mapping 
        return [self.process_one(o) for o in items]
    def process_one(self, item):
        return self.otoi[item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deprocess_one(idx) for idx in idxs]
    def deprocess_one(self, idx): 
        return self.vocab[idx]

###Transform helper functions
# def to_byte_tensor(item):
#     result = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes())) #convert to bytes using PyTorch
#     w,h = item.size
#     return result.view(h,w,-1).permute(2,0,1) #matrix transform to move Channel (back in PIL) to front (front in Pytorch Tesnsors)
# to_byte_tensor._order = 20


# def to_float_tensor(item):
#     _order = 30
#     return item.float().div_(255.) #in range [0,1]
# to_float_tensor._order = 30

# def make_rgb(item):
#     return item.convert('RGB')

# class Transform():
#     _order = 0
    
    
# class convert_to_RGB(Transform):
#     def __call__(self, item):
#         if not isinstance(type(item), type(PIL.Image.Image)):
#             raise TypeError('Images must be of type:' + str(PIL.Image.Image))
#         return item.convert('RGB')

# class ResizedFixed(Transform):
#     _order = 10 #ensure it happens after the other transforms
#     def __init__(self, size):
#             if isinstance(size, int): #make the 2D square dimensions 
#                 size =(size,size)
#             self.size = size
#     def __call__(self, item, image_interpolation=PIL.Image.BILINEAR):
#         if not isinstance(item, PIL.Image.Image):
#             raise TypeError
#         return item.resize(self.size, image_interpolation)

class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics = convert_to_list(metrics)
        self.in_train = in_train
        
    def reset(self):
        self.total_loss = 0.
        self.count = 0
        self.total_metrics = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): 
        return [self.total_loss.item()] + self.total_metrics
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        batch_size = run.xb.shape[0]
        self.total_loss += run.loss * batch_size
        self.count += batch_size
        for i,m in enumerate(self.metrics):
            self.total_metrics[i] += m(run.pred, run.yb) * batch_size



class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

#Hooks -------------------------------------------------------------------------------------------------------------------------------------------
#hook functions

#updated for Leaky Relu 
def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    means.append(outp.data.mean().cpu())
    stds .append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(40,-7,7))

#prints the means and stds of hooked layers
def append_stat(hook, model, input, output):
    d = output.data
    hook.mean, hook.std = d.mean().item(), d.std().item()


###hook helper functions
##applies LSUV initalization to passed in convolutional module layers [Conv1d,2D,3D, linear, ReLU]
#use get_all_modules to get the list of modules to pass in
#model passed in must be the same as where the list of modules comes from
#use get_one_batch -> x_mb is the training minibatch X of the model
def lsuv_module(model, module, x_mb):
    error_ceiling = 1e-3
    hook = ForwardHook(module, append_stat)
    while model(x_mb) is not None and abs(hook.mean) > error_ceiling: #correct the means
        module.bias -= hook.mean
    while model(x_mb) is not None and abs(hook.std - 1) > error_ceiling: #correct the standard deviations
        module.weight.data /= hook.std
    hook.remove()
    return hook.mean, hook.std

#hook classes
#numpy style object container
#used to register a forward hook 
class ForwardHook():
    def __init__(self, m, f): 
        self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


# class ListContainer():
#     def __init__(self, items): self.items = convert_to_list(items)
#     def __getitem__(self, idx):
#         if isinstance(idx, (int,slice)): return self.items[idx]
#         if isinstance(idx[0],bool):
#             assert len(idx)==len(self) # bool mask
#             return [o for m,o in zip(idx,self.items) if m]
#         return [self.items[i] for i in idx]
#     def __len__(self): return len(self.items)
#     def __iter__(self): return iter(self.items)
#     def __setitem__(self, i, o): self.items[i] = o
#     def __delitem__(self, i): del(self.items[i])
#     def __repr__(self):
#         res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
#         if len(self)>10: res = res[:-1]+ '...]'
#         return res

# class ItemList(ListContainer):
#     def __init__(self, items, path='.', tfms=None):
#         super().__init__(items)
#         self.path,self.tfms = Path(path),tfms

#     def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
    
#     def new(self, items, cls=None):
#         if cls is None: cls=self.__class__
#         return cls(items, self.path, tfms=self.tfms)
    
#     def  get(self, i): return i
#     def _get(self, i): return compose(self.get(i), self.tfms)
    
#     def __getitem__(self, idx):
#         res = super().__getitem__(idx)
#         if isinstance(res,list): return [self._get(o) for o in res]
#         return self._get(res)
    

# class ImageList(ItemList):
#     @classmethod
#     def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
#         if extensions is None: extensions = list_image_ext()
#         return cls(get_all_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
#     def get(self, fn): return PIL.Image.open(fn)


# def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path)


#scheduling functions
def combine_schedules(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + convert_to_list(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

def annealer(f): #decorator used to define anneling functions, that can be passed to the ParameterSchedule set_params()
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def linear_scheduler(start, end, position):
    return start+position*(end-start)

#cosine annealing from this paper: https://arxiv.org/pdf/1608.03983.pdf
@annealer
def cosine_scheduler(start, end, position):
    return start + (1 + math.cos(math.pi*(1-position))) * (end-start) / 2

@annealer
def exponential_scheduler(start, end, position):
    return start * (end/start) ** position

@annealer
def no_scheduler(start, end, position):
    return start

    
#hooks container
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

#Generic Optimzier ------------------------------------------------------------
# class Optimizer():
#     def __init__(self, parameters, steppers, **defaults):
#         self.param_groups = list(parameters) #each p_g has its own set of hyperparameters
#         #ensure parameter groups are a list of a list of parameters
#         if not isinstance(self.param_groups, list): self.param_groups = [self.param_groups]
#         self.hyper_params = [{**defaults} for param in self.param_groups] #dictionary to store all hyper parameters, per p_g
#         self.steppers = convert_to_list(steppers)
    
#     #goes through each parameter, in every parameter group
#     def grad_params(self):
#         return [(p, hyper) for pg, hyper in zip(self.param_groups, self.hyper_params) 
#                for p in pg if p.grad is not None] #return all valid gradients
    
#     def zero_grad(self):
#         for param,hyperparam in self.grad_params():
#             param.grad.detach_() #remove from computational graph
#             param.grad.zero_() #zero out
            
#     def step(self):
#         for param,hyper in self.grad_params():
#             compose(param, self.steppers, **hyper)
def get_defaults(default): return getattr(default, '_defaults', {}) #grab default attribute

def _update(os, dest, func):
    for o in os:
        for k,v in func(o).items():
            if k not in dest: #update if missing from destination
                dest[k] = v

class Optimizer():
    def __init__(self, parameters, steppers, **defaults):
        self.steppers = convert_to_list(steppers)
        _update(self.steppers, defaults, get_defaults)
        self.param_groups = list(parameters) #each p_g has its own set of hyperparameters
        #ensure parameter groups are a list of a list of parameters
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups] #dictionary to store all hyper parameters, per p_g
    
    #goes through each parameter, in every parameter group
    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]#return all valid gradients
    
    def zero_grad(self):
        for param,hyperparam in self.grad_params():
            param.grad.detach_() #remove from computational graph
            param.grad.zero_() #zero out
            
    def step(self):
        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)

class StateOptimizer(Optimizer):
    def __init__(self, params, steppers, stats = None, **defaults):
        self.stats = convert_to_list(stats)
        _update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {} #empty dict to track state
    
    def step(self):
        for param, hyper in self.grad_params():
            if param not in self.state:
                self.state[param] = {} #initalize the state, by parameter, if not present
                _update(self.stats, self.state[param], lambda o: o.init_state(param))
            state = self.state[param]
            for stat in self.stats: #update the state
                state = stat.update(param, state, **hyper)
            compose(param, self.steppers, **state, **hyper) #recompose stepper functions
            self.state[param] = state
        
class Stats():
    _defaults = {}
    def init_state(self, param): raise NotImplementedError #initalize state of the state
    def update(self, param, state, **kwargs): raise NotImplementedError #update stat's state
        
#average gradient state
class AverageGradient(Stats):
    _defaults = dict(momentum=0.9)    
    def init_state(self, param):
        return {'gradient_avg': torch.zeros_like(param.grad.data)}
    
    def update(self, param, state, momentum, **kwargs):
        state['gradient_avg'].mul_(momentum).add_(param.grad.data)
        return state
## Optimizer Steppers - to-do convert monkey-patched defaults into annotated defaults
def sgd_stepper(param, learning_rate, **kwargs): 
    """performs the SGD step"""
    param.data.add_(-learning_rate, param.grad.data)
    return param

def weight_decay_stepper(param, learning_rate, weight_decay, **kwargs):
    param.data.mul_(1 - learning_rate * weight_decay)
    return param
weight_decay_stepper._defaults = dict(weight_decay=0.)

def l2_regularization_stepper(param, learning_rate, weight_decay, **kwargs):
    param.grad.data.add_(weight_decay, param.data)
    return param
l2_regularization_stepper._defaults = dict(weight_decay=0.1)

def adam_stepper(p, learning_rate, mom, mom_damp, step_count, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
#     adam_stepper._defaults = dict(eps=1e-5)
    debias1 = debias(mom,     mom_damp, step_count)
    debias2 = debias(sqr_mom, sqr_damp, step_count)
    p.data.addcdiv_(-learning_rate / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
    return p
adam_stepper._defaults = dict(eps=1e-5)

def lamb_step(p, learning_rate, mom, mom_damp, step_count, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, weight_decay, **kwargs):
    debias1 = debias(mom,     mom_damp, step_count)
    debias2 = debias(sqr_mom, sqr_damp, step_count)
    r1 = p.data.pow(2).mean().sqrt()
    step_count = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + weight_decay*p.data
    r2 = step_count.pow(2).mean().sqrt()
    p.data.add_((-learning_rate * min(r1/r2,10)), step_count)
    return p
lamb_step._defaults = dict(eps=1e-6, weight_decay=0.)

#Callbacks ---------------------------------------------------------------------------------------------------------------------------------------
#Optimizer Changes to Callbacks: only thing that is different is hyperparameters are what we use in callbacks instead of just param_groups
#                   if using torch.optim -> execute by going through torch.optim.param_groups
#                   if using Generalized optimizer -> execute by going through each hyper-parameter
# class Callback():
#     _order = 0
#     def set_runner(self, run): self.run=run
#     def __getattr__(self, k): return getattr(self.run, k)

#     @property
#     def name(self):
#         name = re.sub(r'Callback$', '', self.__class__.__name__)
#         return camel2snake(name or 'callback')

#     def __call__(self, cb_name):
#         f = getattr(self, cb_name, None)
#         if f and f(): return True
#         return False

# class Recorder(Callback):
#     def begin_fit(self): self.lrs,self.losses = [],[]

#     def after_batch(self):
#         if not self.in_train: return
#         self.lrs.append(self.opt.hypers[-1]['learning_rate']) #goes through hyper parameters
#         self.losses.append(self.loss.detach().cpu())        

#     def plot_lr  (self): plt.plot(self.lrs)
#     def plot_loss(self): plt.plot(self.losses)
        
#     def plot(self, skip_last=0):
#         losses = [o.item() for o in self.losses]
#         n = len(losses)-skip_last
#         plt.xscale('log')
#         plt.plot(self.lrs[:n], losses[:n])


# class ParamScheduler(Callback):
#     _order=1
#     def __init__(self, pname, sched_funcs):
#         self.pname,self.sched_funcs = pname,convert_to_list(sched_funcs)

#     def begin_batch(self): 
#         if not self.in_train: return
#         fs = self.sched_funcs
#         if len(fs)==1: fs = fs*len(self.opt.param_groups)
#         pos = self.n_epochs/self.epochs
#         for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)
            
# class LR_Find(Callback):
#     _order=1
#     def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
#         self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
#         self.best_loss = 1e9
        
#     def begin_batch(self): 
#         if not self.in_train: return
#         pos = self.n_iter/self.max_iter
#         lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
#         for pg in self.opt.hypers: pg['learning_rate'] = lr
            
#     def after_step(self):
#         if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
#             raise CancelTrainException()
#         if self.loss < self.best_loss: self.best_loss = self.loss


# ###These Callbacks are based on torch.optim, look above for the Callbacks based on the generalized Optimizer
# #our standard used callback, included in every model
# class TrainEvalCallback(Callback):
#     def begin_fit(self):
#         self.run.n_epochs=0.
#         self.run.n_iter=0

#     def after_batch(self):
#         if not self.in_train: return
#         self.run.n_epochs += 1./self.iters
#         self.run.n_iter   += 1
        
#     def begin_epoch(self):
#         self.run.n_epochs=self.epoch
#         self.model.train()
#         self.run.in_train=True

#     def begin_validate(self):
#         self.model.eval()
#         self.run.in_train=False

# class AvgStatsCallback(Callback):
#     def __init__(self, metrics):
#         self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
        
#     def begin_epoch(self):
#         self.train_stats.reset()
#         self.valid_stats.reset()
        
#     def after_loss(self):
#         stats = self.train_stats if self.in_train else self.valid_stats
#         with torch.no_grad(): stats.accumulate(self.run)
    
#     def after_epoch(self):
#         print(self.train_stats)
#         print(self.valid_stats)
        
# class CudaCallbackDev(Callback):
#     def __init__(self,device): self.device=device
#     def begin_fit(self): 
#         self.model.to(self.device)
#     def begin_batch(self):
#         self.run.xb,self.run.yb = self.xb.to(self.device),self.yb.to(self.device)

# class CudaCallback(Callback):
#     def begin_fit(self): self.model.cuda()
#     def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

# #applies a given transform to the independent variables (xs)
# class IndependentVarBatchTransformCallback(Callback):
#     _order=2
#     def __init__(self, tfm): self.tfm = tfm
#     def begin_batch(self): self.run.xb = self.tfm(self.xb)


# class Recorder(Callback):
#     def begin_fit(self):
#         self.lrs = [[] for _ in self.opt.param_groups]
#         self.losses = []

#     def after_batch(self):
#         if not self.in_train: return
#         for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
#         self.losses.append(self.loss.detach().cpu())        

#     def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
#     def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
        
#     def plot(self, skip_last=0, pgid=-1):
#         losses = [o.item() for o in self.losses]
#         lrs    = self.lrs[pgid]
#         n = len(losses)-skip_last
#         plt.xscale('log')
#         plt.plot(lrs[:n], losses[:n])

# class ParamScheduler(Callback):
#     _order=1
#     def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs
        
#     def begin_fit(self):
#         if not isinstance(self.sched_funcs, (list,tuple)):
#             self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

#     def set_param(self):
#         assert len(self.opt.param_groups)==len(self.sched_funcs)
#         for pg,f in zip(self.opt.param_groups,self.sched_funcs):
#             pg[self.pname] = f(self.n_epochs/self.epochs)
            
#     def begin_batch(self): 
#         if self.in_train: self.set_param()

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = convert_to_list(cbs)
        for cbf in convert_to_list(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs


    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_function(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl: self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, mod, optimizer, loss_func, train_dl, valid_dl):
        self.epochs = epochs
        self.loss = tensor(0.)
        self.model = mod
        self.opt = optimizer
        self.loss_function = loss_func
        self.valid_dl = valid_dl
        self.train_dl = train_dl
        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.valid_dl)
                self('after_epoch')
            
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        return_val = False
        for cb in sorted(self.cbs, key=lambda x: x._order): return_val = cb(cb_name) or return_val
        return return_val



#Pytorch Module
#helper function to summarize models, architectures, schedules, parameters, ect..
def model_summary(runner, model, valid_dl, find_all=False):
    runner.in_train = False #monkey patch to getaround callback ordering issues
    xb,yb = get_one_batch(valid_dl, runner)
    device = next(model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(model, is_lin_layer) if find_all else model.children() #find the linear layers, or the immediate children
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks: model(xb)

#helper function to find modules within a model recursively
def get_all_modules(model, predictate):
    if predictate(model):
        return [model]
    return sum([get_all_modules(o, predictate) for o in model.children()], [])

#passed as predicate function to get_all_modules
def is_linear_layer(layer):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(layer, lin_layers)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

class GeneralReLU(nn.Module): #generalized ReLU class
    def __init__(self, leak=None, sub_value=None, value_cuttoff=None):
        super().__init__()
        self.leak = leak
        self.sub = sub_value
        self.cuttoff = value_cuttoff
        
    def forward(self, x_in):
        x_in = F.leaky_relu(x_in, self.leak) if self.leak is not None else F.relu(x_in)
        if self.sub is not None:
            x_in.sub_(self.sub)
        if self.cuttoff is not None:
            x_in.clamp_max_(self.cuttoff)
        return x_in

##use nn.BatchNorm2d(), this was built for display of ability    
class Batch_Normalization(nn.Module):
    def __init__(self, nf, momentum=0.1, epsilon=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.momentum = momentum
        self.epsilon = epsilon
        self.multipliers = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('variances',  torch.ones(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))

    def update_stats(self, x):
        m = x.mean((0,2,3), keepdim=True) #averaging over batches,x-coordinates,y-coordinates
        v = x.var((0,2,3), keepdim=True) #averaging over batches,x-coordinates,y-coordinates
        self.means.lerp_(m, self.momentum)
        self.variances.lerp_ (v, self.momentum)
        return m,v
        
    def forward(self, x):
        if self.training:
            with torch.no_grad(): m,v = self.update_stats(x)
        else: m,v = self.means,self.variances
        x = (x-m) / (v+self.epsilon).sqrt()
        return x*self.multipliers + self.adds       


class ConvLayer2D(nn.Module):
    def __init__(self, num_in, num_fm, kernel_size = 3, stride=2, sub=0., **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_fm, kernel_size, padding=kernel_size//2, stride=stride, bias=True)
        self.relu = GeneralReLU(sub_value=sub, **kwargs)

    def forward(self, x):
         return self.relu(self.conv(x))

    @property
    def bias(self): return -self.relu.sub

    @property
    def weight(self): return self.conv.weight
    
    @bias.setter
    def bias(self, value): self.relu.sub = -value

    