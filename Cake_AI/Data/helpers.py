from fastai import datasets
from collections import OrderedDict
import pickle 
import gzip 
import os
import re
from pathlib import Path
from typing import Iterable
from torch import tensor
from Data.Dataset import ItemList

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



#splits by grandparent directory using the passed in parent directory names
def grandparent_splitter(fn, valid_name='valid', train_name='train'):
    gp = fn.parent.parent.name
    return True if gp==valid_name else False if gp==train_name else None

#splits a list of items by a given function
def split_by_function(items, func):
    mask = [func(o) for o in items]
    # `None` values will be filtered out
    f_itms = [o for o,m in zip(items,mask) if m==False]
    t_itms = [o for o,m in zip(items,mask) if m==True ]
    return f_itms,t_itms

def parent_labeler(fn): 
    return fn.parent.name


def get_unique_keys(lst, sort=False):
    result = list(OrderedDict.fromkeys(lst).keys()) #get the keys
    if sort is True: result.sort()
    return result
