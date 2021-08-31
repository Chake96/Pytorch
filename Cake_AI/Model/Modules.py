from torch import nn, no_grad, torch
from ..Data.helpers import get_one_batch

def model_summary(runner, model, valid_dl, find_all=False):
    runner.in_train = False #monkey patch to getaround callback ordering issues
    xb,yb = get_one_batch(valid_dl, runner)
    device = next(model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = get_all_modules(model, is_linear_layer) if find_all else model.children() #find the linear layers, or the immediate children
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
