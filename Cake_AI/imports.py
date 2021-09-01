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
