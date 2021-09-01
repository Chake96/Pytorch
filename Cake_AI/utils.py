from .Data.helpers import convert_to_list

#composes a list of functions together
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(convert_to_list(funcs), key=key): x = f(x, **kwargs)
    return x

def get_data(path_in, encoding_in='latin-1'):
    path = datasets.download_data(path_in, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding=encoding_in)
    return map(tensor, (x_train,y_train,x_valid,y_valid))

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

def camel2snake(name):
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

#normalize datasets
def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

def normalize_channels(x, mean, std):
    return (x-mean[...,None,None]) / std[...,None,None]

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
