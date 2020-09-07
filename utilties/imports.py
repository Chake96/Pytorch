from fastai import datasets
from pathlib import Path
from IPython.core.debugger import set_trace
import pickle, gzip, math, torch, re, matplotlib as mpl, matplotlib.pyplot as plt
from functools import partial

from typing import Iterable
from torch import tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset,  SequentialSampler, RandomSampler
import torch.nn.functional as F

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


#used to implement extra functionallity into our training
class Callback():
    _order=0
    def set_runner(self, runner): self.run=runner
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

#callback included in every runner, tracks iterations and epochs
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

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics,True)
        self.valid_stats = AvgStats(metrics,False)
        
    def begin_epoch(self):        
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)

#used to define a model and train it
class Runner():
    def __init__(self, cbs=None, cb_functions=None):
        callbacks = convert_to_list(cbs)
        for cbf in convert_to_list(cb_functions): #convert functions to callbacks
            cb = cbf()
            setattr(self, cb.name, cb)
            callbacks.append(cb)
        self.stop = False
        self.callbacks = [TrainEvalCallback()]+callbacks

    def one_batch(self, xb, yb):
        self.xb,self.yb = xb,yb
        if self('begin_batch'): return
        self.pred = self.model(self.xb)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb,yb in dl:
            
            if self.stop: break
            self.one_batch(xb, yb)
            self('after_batch')
        self.stop=False

    #the new training loop
    def fit(self, epochs_in, model_in, optimizer_in, loss_function_in, train_dl_in, valid_dl_in):
        self.epochs = epochs_in
        self.model = model_in
        self.opt = optimizer_in
        self.loss_func = loss_function_in
        self.train_dl = train_dl_in
        self.valid_dl = valid_dl_in
        try:
            for cb in self.callbacks: cb.set_runner(self)
            if self('begin_fit'): return
            for epoch in range(self.epochs):
                self.epoch = epoch
                #training
                if not self('begin_epoch'): #callbacks default to false returns
                    self.all_batches(self.train_dl)
                
                #validation 
                with torch.no_grad(): 
                    if not self('begin_validate'): 
                        self.all_batches(self.valid_dl)
                if self('after_epoch'): break
            
        finally:
            self('after_fit')
            self.model = None
            self.opt = None
            self.loss_func = None
            self.train_dl = None
            self.valid_dl = None

    def __call__(self, cb_name):
        for cb in sorted(self.callbacks, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f():
                return True
        return False
    