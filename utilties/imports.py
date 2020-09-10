from fastai import datasets
from pathlib import Path
from IPython.core.debugger import set_trace
import pickle, gzip, math, torch, re, matplotlib as mpl, matplotlib.pyplot as plt
from functools import partial
import numpy
from typing import Iterable
from torch import tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset,  SequentialSampler, RandomSampler
import torch.nn.functional as F
import torch.nn.init

#helper functions
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

def accuracy(out, yb): #
     return (torch.argmax(out, dim=1)==yb).float().mean()

def normalize(x, m, s):
    return (x-m)/s

#normalize datasets
def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

#resize mnist image data -> 28x28
def mnist_resize(x): return x.view(-1, 1, 28, 28)

#flatten an image to 
def flatten(x):
    return x.view(x.shape[0], -1) #removes 1,1 axis from result of AvgPool layer

#re-view an x variable at a specific size
def view_tfm(*size):
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

#creating 2d convolution models
def conv2d(ni, nf, ks=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), nn.ReLU())

def get_cnn_layers(num_categories, nfs):
    nfs = [1] + nfs
    return [
        conv2d(nfs[i], nfs[i+1], 5 if i==0 else 3) for i in range(len(nfs)-1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], num_categories)]

def get_cnn_model(num_categories, nfs):
    return nn.Sequential(*get_cnn_layers(num_categories, nfs))

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
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)