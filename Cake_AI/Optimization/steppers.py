def debias(mom, damp, step): 
    return damp * (1 - mom**step) / (1-mom)

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

