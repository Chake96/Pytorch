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

def accuracy(out, yb): #
     return (torch.argmax(out, dim=1)==yb).float().mean()

def normalize(x, m, s):
    return (x-m)/s

#normalize datasets
def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

#image dataset helper functions
#resize mnist image data -> 28x28
def mnist_resize(x): return x.view(-1, 1, 28, 28)

#flatten an image to 
def flatten(x):
    return x.view(x.shape[0], -1) #removes 1,1 axis from result of AvgPool layer

#re-view an x variable at a specific size
def view_tfm(*size):
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

def list_image_ext():
    return set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

#statistics helper functions
#updated for Leaky Relu 
def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    means.append(outp.data.mean().cpu())
    stds .append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(40,-7,7))

#hook for mean and std
def append_mean_std(hook, model, inp, outp):
    if not hasattr(hook, 'stats'):
        hook.stats=([],[])
    means, stds = hook.stats
    if model.training:
        means.append(outp.data.mean())
        stds.append(outp.data.std())



#creating 2d convolution models
#passing in GeneralReLU args
def get_cnn_layers(num_categories, num_features, layer, **kwargs):
    num_features = [1] + num_features
    return [layer(num_features[i], num_features[i+1], 5 if i==0 else 3, **kwargs)
            for i in range(len(num_features)-1)] + [
        nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(num_features[-1], num_categories)]

#dont need bias, is using batchnorm
def conv_layer(ni, num_features, ks=3, stride=2, batch_norm=True, **kwargs):
    layers = [nn.Conv2d(ni, num_features, ks, padding=ks//2, stride=stride, bias = not batch_norm), 
            GeneralReLU(**kwargs)]
    if batch_norm:
        layers.append(torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1))
        # layers.append(Batch_Normalization(num_features))
    return nn.Sequential(*layers)

#0s all the bias weights, calls the passed initalizations function on each layer in the model
def init_cnn_(model, func):
    if isinstance(model, nn.Conv2d):
        func(model.weight, a =0.1)
        if getattr(model, 'bias', None) is not None:
            model.bias.data.zero_()
    for layer in model.children():
        init_cnn_(layer, func)

def init_cnn(model, uniform=False):
    f = torch.nn.init.kaiming_uniform_ if uniform else torch.nn.init.kaiming_normal_
    init_cnn_(model, f)
            
def get_cnn_model(num_categories, num_features, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(num_categories, num_features, layer, **kwargs))

#returns the model and optimizer, will be refactored to be more versatile, currently just used for simple testing
def get_model(training_dl, lr=0.5, nh=50):
    m = training_dl.dataset.x_dataset.shape[1]
    categories = training_dl.dataset.y_dataset.max().item()+1
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,categories))
    return model, optim.SGD(model.parameters(), lr=lr)

def get_data(path_in, encoding_in='latin-1'):
    path = datasets.download_data(path_in, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding=encoding_in)
    return map(tensor, (x_train,y_train,x_valid,y_valid))

#takes a batch from a dataloader and applies the supplied runner callbacks to it
def get_one_batch(dl, runner):
    runner.xb, runner.yb = next(iter(dl))
    for cb in runner.cbs:
        cb.set_runner(runner)
    runner('begin_batch')
    return runner.xb, runner.yb

class Dataset():
    def __init__(self, x_ds, y_ds):
        self.x_dataset = x_ds
        self.y_dataset = y_ds
    
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, i):
        return self.x_dataset[i],self.y_dataset[i]

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


class ListContainer():
    def __init__(self, items): self.items = convert_to_list(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

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


#Callbacks ---------------------------------------------------------------------------------------------------------------------------------------
class Callback():
    _order = 0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

#our standard used callback, included in every model
class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
        
    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
        
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)
        
class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())        

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
        
    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs
        
    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)
            
    def begin_batch(self): 
        if self.in_train: self.set_param()

class CudaCallbackDev(Callback):
    def __init__(self,device): self.device=device
    def begin_fit(self): 
        self.model.to(self.device)
    def begin_batch(self):
        self.run.xb,self.run.yb = self.xb.to(self.device),self.yb.to(self.device)

class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

#applies a given transform to the independent variables (xs)
class IndependentVarBatchTransformCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.xb)
            


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

    