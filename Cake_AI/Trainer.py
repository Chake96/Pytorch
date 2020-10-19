from .Exceptions import *
from .Data.helpers import convert_to_list
from .Callbacks.Callbacks import TrainEvalCallback
from .Model.2D_CNN import get_cnn_model

def create_model_trainer(nfs, num_ch, num_cat, train_dl, valid_dl, in_layer, cbs_in=None, **kwargs):
    model = get_cnn_model(train_dl, valid_dl, num_ch, num_cat, nfs, in_layer, **kwargs)
    init_cnn(model)
#     return get_runner(model, data, lr=lr, cbs=cbs_in, opt_func=opt_func)
    return model, Trainer(cb_funcs=cbs_in)
 

class Trainer():
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
