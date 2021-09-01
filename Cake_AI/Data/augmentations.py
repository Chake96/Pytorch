import mimetypes
import PIL
import torch
import random
from .Dataset import ImageList
from pathlib import Path

#image helper functions-------------------------------------------------------
def list_image_ext():
    """lists the image extensions available on the current system"""
    return set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))


#flatten an image to 
def flatten(x):
    """removes 1,1 axis from result, typically used in Average Pooling Layer"""
    return x.view(x.shape[0], -1)


def to_byte_tensor(item):
    """converts passed item argument to a torch tensor"""
    result = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes())) #convert to bytes using PyTorch
    w,h = item.size
    return result.view(h,w,-1).permute(2,0,1) #matrix transform to move Channel (back in PIL) to front (front in Pytorch Tesnsors)
to_byte_tensor._order = 20


def to_float_tensor(item):
    _order = 30
    return item.float().div_(255.) #in range [0,1]
to_float_tensor._order = 30

def make_rgb(item):
    return item.convert('RGB')


def two_step_resize(image, x_size, y_size):
    """takes a path or PIL.Image and returns it after a bicubic and nearest neighbor resizing using PIL.resize"""
    size = (x_size, y_size)
    ret_img = PIL.Image
    if isinstance(image, PIL.Image.Image):
        ret_img = image.resize(size, resample=PIL.Image.BICUBIC).resize(size, resample = PIL.Image.NEAREST)
    elif isinstance(image, Path):
        img = PIL.Image.open(image)
        ret_img = two_step_resize(img, x_size, y_size)
    else:
        raise TypeError("image must be a Pathlib.Path or PIL.Image.Image")
    return ret_img
def PIL_random_flip(x):
    return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random() < 0.5 else x

def get_image_list(tfms, dataset_path=Path('/')):
    return ImageList.from_files(dataset_path, tfms=tfms)


#image transforms
class Transform():
    """base class for defining tranforms; default _order is 0"""
    _order = 0
    
    
class convert_to_RGB(Transform):
    def __call__(self, item):
        if not isinstance(type(item), type(PIL.Image.Image)):
            raise TypeError('Images must be of type:' + str(PIL.Image.Image))
        return item.convert('RGB')

class ResizedFixed(Transform):
    _order = 10 #ensure it happens after the other transforms
    def __init__(self, size):
            if isinstance(size, int): #make the dimensions a tuple for a 2D square 
                size = (size,size)
            self.size = size
    def __call__(self, item, image_interpolation=PIL.Image.BILINEAR):
        if not isinstance(item, PIL.Image.Image):
            raise TypeError
        return item.resize(self.size, image_interpolation)

class PIL_Transform(Transform):
    """a base class for PIL Data Augmentations, _order: 11"""
    _order=11 #give all PIL transforms the same order

class PIL_Rand_Flip(PIL_Transform):
    """take a probability as input, returns the image with PIL.Image.FLIP_LEFT_RIGHT applied if the threshold is met"""
    _order = 11
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        return img.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random() < self.p else img
    
class PIL_Rand_Dihedral(PIL_Transform):
    def __init__(self, p=0.75): 
        self.p=p*7/8 #Little hack to get the 1/8 identity dihedral transform taken into account.
    def __call__(self, x):
        if random.random()>self.p: return x
        return x.transpose(random.randint(0,6))