from .Data.helpers import convert_to_list
from utils import compose

def get_defaults(default): 
    return getattr(default, '_defaults', {}) #grab default attribute

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