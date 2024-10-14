import numpy as np
from sklearn.decomposition import PCA
from Lattice_type import *
from numpy.random import rand, randn,  randint
import PIL
from PIL import Image
import matplotlib.pyplot as plt


def obj_alloc(num):
    if 'int' in f'{type(num)}':
        return Integer(num)
    elif 'float' in f'{type(num)}':
        return Float(num)
    else:
        return array(num)


class image_loader():
    def __init__(self,img):
        self.image = Image.open(img) if type(img) == str else img

    def numpy(self,):
        return to_numpy(self.image, dt=np.int32)
    
    def lattice(self):
        return array(self.numpy())
    
    def resize(self,size:tuple):
        self.image = self.image.resize(size)
        return image_loader(self.image)
    
    def convert(self,prop):
        return image_loader(self.image.convert(prop))
    
    def __repr__(self):
        return f"{self.image}"
    
def to_numpy(val,ar_type=0,dt=np.float32,):
    if ar_type == 0:
        arr = np.asarray(val,dtype=dt)
    elif ar_type == 1:
        arr = np.array(val,dtype=dt)
    else:
        raise Exception('Array type can either be 0 (np.asarray) or 1 (np.array)')
    return arr
    
def as_array(val,requires_grad=False):
    return array(val,requires_grad=requires_grad)
        
def arange(*args,requires_grad=False):
    val = np.arange(*args)
    return array(val,requires_grad=requires_grad)

def ones(*args,requires_grad=False):
    val = np.ones(*args)
    return array(val,args,requires_grad=requires_grad)

def zeros(*args,requires_grad=False):
    val = np.zeros(*args)
    return array(val,args,requires_grad=requires_grad)


def eye(*args,requires_grad=False):
    val = np.eye(*args)
    args = (args[0],args[0]) if len(args) == 1 else args
    return array(val,args,requires_grad=requires_grad)

def rand(*args,requires_grad=False):
    val = np.random.rand(*args)
    return array(val,args,requires_grad=requires_grad)

def randn(*args,requires_grad=False):
        val= np.random.randn(*args)
        return array(val,args,requires_grad=requires_grad)
    
def linspace(*args,requires_grad=False):
    val = np.linspace(*args)
    return array(val,args[2:],requires_grad=requires_grad)
    
def randint(*args,requires_grad=False):
    val= np.random.randint(*args)
    return array(val,args[2],requires_grad=requires_grad)


def zeros_like(data,requires_grad=False):
    data = to_numpy(data)
    val = np.zeros_like(data)
    return array(val,requires_grad=requires_grad)


def ones_like(data,requires_grad=False):
    data = to_numpy(data)
    val = np.ones_like(data)
    return array(val,requires_grad=requires_grad)



