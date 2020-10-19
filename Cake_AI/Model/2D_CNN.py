from torch import nn
from .Modules import GeneralReLU, Lambda
from ..Data.augmentations import flatten
from ..math_utils import prev_pow_2
#helper functions for manipulating and creating Pytorch 2D convolutional models

#creating 2d convolution models
#passing in GeneralReLU args
# def get_cnn_layers(num_categories, num_features, layer, **kwargs):
#     num_features = [1] + num_features
#     return [layer(num_features[i], num_features[i+1], 5 if i==0 else 3, **kwargs)
#             for i in range(len(num_features)-1)] + [
#         nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(num_features[-1], num_categories)]

#doesnt need bias, is using batchnorm
def conv_layer(ni, num_features, ks=3, stride=2, batch_norm=True, **kwargs):
    """creates a 2D convolutional layer with a GeneralRelu Layer, optional: torch.nn.BatchNorm2d layer"""
    layers = [nn.Conv2d(ni, num_features, ks, padding=ks//2, stride=stride, bias = not batch_norm), 
            GeneralReLU(**kwargs)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1))
        # layers.append(Batch_Normalization(num_features))
    return nn.Sequential(*layers)

#0s all the bias weights, calls the passed initalizations function on each layer in the model
def init_cnn_(model, func):
    if isinstance(model, nn.Conv2d):
        func(model.weight, a =0.1)
        if getattr(model, 'bias', None) is not None:
            model.bias.data.zero_()
    else:
        raise TypeError("Model parameter must be: Torch.nn.Conv2d")
    for layer in model.children():
        init_cnn_(layer, func)

def init_cnn(model, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(model, f)




def get_cnn_layers(train_dl, valid_dl, num_ch, num_cat, nfs,layer, **kwargs):
    """Creates Pytorch layers for a basic 2D-CNN Model Classifier

    Args:
        train_dl - a Pytorch Dataloader for the Training Dataset
        valid_dl - a Pytorch Dataloader for the Validation Dataset
        num_ch - the number of channels for the first input layer
        num_cat - the number of categories the last layer outputs; eg: MNIST has 10 categories
        nfs - the feature exchange dimensions between hidden layers, passed as a list
        layer - the type of Pytorch Layer to create

    Returns:
        a list of layers with the following key characteristics:
            a)the first layer takes num_ch as the input size and has an output size of the previous power of num_ch*3*3, the  3*3 represents the kernal size
            b)hidden layers with appropriately scaled input/output sizes based on the nfs argument 
            c)the final layer: Pooling -> Flatten -> Linear
                note: the linear layer has output size equal to the number of categories (num_cat)
    """
    def f(ni, nf, stride=2): 
        return layer(ni, nf, 3, stride=stride, **kwargs)
    l2 = prev_pow_2(num_ch*3*3)
    #3x3 kernel sizes
    layers =  [f(num_ch  , l2  , stride=1),
               f(l2  , l2*2, stride=2),
               f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]  #build the layers with proper input/output sizes
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten),  #a typical last layer, has num_categories channels out
               nn.Linear(nfs[-1], num_cat)]
    return layers
            


def get_cnn_model(train_dl, valid_dl, num_chs, num_cat, nfs, layer, **kwargs):
    """Builds a Sequential Model built with get_cnn_layers()
        
        Args:
            train_dl - a Pytorch Dataloader for the Training Dataset
            valid_dl - a Pytorch Dataloader for the Validation Dataset
            num_ch - the number of channels for the first input layer
            num_cat - the number of categories the last layer outputs; eg: MNIST has 10 categories
            nfs - the feature exchange dimensions between hidden layers, passed as a list
            layer - the type of Pytorch Layer to create
        
        Returns:
            a Sequential 2D CNN Model created by calling the get_cnn_layers helper function
    """

    return nn.Sequential(*get_cnn_layers(train_dl, valid_dl, num_chs, num_cat, nfs, layer, **kwargs))