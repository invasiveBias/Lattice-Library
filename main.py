import numpy as np
from sklearn.decomposition import PCA
from Lattice_type import *
from numpy.random import rand, randn,  randint
import PIL
from PIL import Image
import matplotlib.pyplot as plt




class image_loader():
    def __init__(self,img):
        self.image = Image.open(img) if type(img) == str else img

    def numpy(self,):
        return np.array(self.image, dtype=int)
    
    def lattice(self):
        return array(self.numpy())
    
    def resize(self,size:tuple):
        self.image = self.image.resize(size)
        return image_loader(self.image)
    
    def convert(self,prop):
        return image_loader(self.image.convert(prop))
    
    def __repr__(self):
        return f"{self.image}"
    
    
def as_array(val,requires_grad=False):
    return array(val,requires_grad=requires_grad)
        
def arange(*args,requires_grad=False):
    val = np.arange(*args)
    return array(val,requires_grad=requires_grad)

def ones(shape, dtype=float,requires_grad=False):
    val = np.ones(shape, dtype,)
    return array(val,requires_grad=requires_grad)

def zeros(shape, dtype=float,requires_grad=False):
    val = np.zeros(shape, dtype)
    return array(val,requires_grad=requires_grad)


def eye(N, M=None, k=0, dtype=float,requires_grad=False):
    val = np.eye(N, M, k, dtype)
    return array(val,requires_grad=requires_grad)

def rand(*args,requires_grad=False):
    val = np.random.rand(*args)
    return array(val,requires_grad=requires_grad)

def randn(*args,requires_grad=False):
        val= np.random.randn(*args)
        return array(val,requires_grad=requires_grad)
    
    
def choice(a, size=None, replace=True, p=None, requires_grad=False):
    val = np.random.choice(a, size=size, replace=replace, p=p)
    return array(val,requires_grad=requires_grad)

    
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0,requires_grad=False):
    val = np.linspace(start, stop, num, endpoint, retstep, dtype, axis)
    return array(val,requires_grad=requires_grad)
    
def randint(low, high=None, size=None, dtype=int,requires_grad=False):
    val= np.random.randint(low,high,size,dtype)
    return array(val,requires_grad=requires_grad,dtype=int)


def zeros_like(a, dtype=None,requires_grad=False):
    data = a._data
    val = np.zeros_like(data, dtype,)
    return array(val,requires_grad=requires_grad)


def ones_like(a, dtype=None,requires_grad=False):
    data = a._data
    val = np.ones_like(data,dtype)
    return array(val,requires_grad=requires_grad)

def empty(shape, dtype=object,requires_grad=False):
    val = np.empty(shape,dtype=dtype)
    return array(val,requires_grad=requires_grad)



def concatenate(lst:list,axis=0,*args,**kwargs):
    n_lst = []
    for i in lst:
        i = array(i) if f"{type(i)}" != 'lattice.array' else i
        n_lst.append(i._data)
    try:
        arr = np.concatenate(n_lst,axis=axis)
    except ValueError as err:
        arr = np.array(n_lst)
    return array(arr)

def column_stack(lst:tuple):
    return array(np.column_stack(lst))

def row_stack(lst:tuple):
    return array(np.row_stack(lst))




