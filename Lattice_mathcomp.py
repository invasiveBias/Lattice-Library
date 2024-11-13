import numpy as np
from numpy.random import rand, randn,  randint
from sklearn.decomposition import PCA
from main import *
from Lattice_type import *



# vector transformations

def sum(val,dim= None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.sum(val._data,axis=dim) 
    return array(arr,requires_grad = val.requires_grad)
    
    
def mean(val,dim=None):
    val = array(val) if type(val) != array else val
    arr = np.mean(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def median(val,dim=None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.median(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def nanmean(val,dim=None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.nanmean(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def nansum(val,dim= None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.nansum(val._data,axis=dim) 
    return array(arr,requires_grad = val.requires_grad)

def norm(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.linalg.norm(val._data)
    return array(arr, requires_grad = val.requires_grad)

def dot(val,val2):
        arr = np.dot(val._data,val2._data)
        requires_grad = val.requires_grad or val2.requires_grad
        return array(arr, requires_grad = requires_grad)
    
def unique(val,return_counts= False):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.unique(val._data,return_counts= return_counts) 
    if return_counts == False:
        return array(arr, requires_grad = val.requires_grad)
    else:
        return array(arr[0], requires_grad = val.requires_grad) , arr[1]


# element-wise transformations

def abs(val,*args):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    val._data = np.abs(val._data,*args)
    return val

def exp(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    val._data = np.exp(val._data.astype(float))
    return val

def log(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    val._data = np.log(val._data.astype(float))
    return val

def log10(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.log10(val._data.astype(float))
    return array(arr)

def log2(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.log2(val._data.astype(float))
    return array(arr)


def sqrt(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    val._data = np.sqrt(val._data.astype(float))
    return val

def argsort(val,*args, **kwargs):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    ind = np.argsort(val._data)
    return array(ind)

def argmax(val,axis=None,*args, **kwargs):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    ind = np.argmax(val._data,axis=axis)
    return array(ind)

def fill_diagonal(val1, val2):
    val1 = as_array(val) if f"{type(val1)}" != 'lattice.array' else val1
    val2 = as_array(val2) if f"{type(val2)}" != 'lattice.array' else val2
    np.fill_diagonal(val1._data,val2._data)
    return val1

def cov(val, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.cov(val._data, y=y, rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights, aweights=aweights, dtype=dtype)
    return array(arr)
    
def where(condition, x, y):
    val = np.where(condition._data, x,y)
    return array(val)

def sign(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    val = np.sign(val._data)
    return array(val)

def bincount(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.bincount(val._data)
    return array(arr) 

def argwhere(val):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.argwhere(val._data)
    return array(arr)

def percentile(val,q,axis=None):
    val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
    arr = np.percentile(val._data,q,axis=axis)
    return array(arr)

def full(shape, fill_value, dtype=None):
    fill_value = array(fill_value) if type(fill_value) != array else fill_value
    arr = np.full(shape=shape, fill_value=fill_value._data,dtype=dtype)
    return array(arr)

class matrix_comp():
    def __init__(Self):
        pass
       
    @staticmethod
    def matmul(val1,val2):
        val1 = as_array(val) if f"{type(val1)}" != 'lattice.array' else val1
        val2 = as_array(val2) if f"{type(val2)}" != 'lattice.array' else val2
        obj = val1 @ val2
        return obj
    
    
    @staticmethod
    def mat_inv(val):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        val._data = np.linalg.inv(val._data)
        return val
    
    @staticmethod
    def eigen(val):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        e_val, e_vec = np.linalg.eig(val._data)
        return array(e_val.real), array(e_vec.real)
    
    @staticmethod
    def det(val):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        val._data = np.linalg.det(val._data)
        return val
    
    @staticmethod
    def svd(val,herm=False):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        u,d,vt = np.linalg.svd(val._data,hermitian=herm)
        return array(u), array(d), array(vt)
    
    @staticmethod
    def pinv(val):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        val._data = np.linalg.pinv(val._data)
        return val
    
    @staticmethod
    def trace(val):
        val = as_array(val) if f"{type(val)}" != 'lattice.array' else val
        val._data = np.trace(val._data)
        return val
    
    
    @staticmethod
    def pca(X,k=2):
        X = as_array(X) if f"{type(X)}" != 'lattice.array' else X
        X_meaned = X - mean(X , dim = 0)
        cov_mat = cov(X_meaned , rowvar = False)
        eigen_val, eigen_vec = matrix_comp.eigen(cov_mat)
        sorted_index = argsort(eigen_val)[::-1]
        sorted_eigenvalue = eigen_val[sorted_index]
        sorted_eigenvectors = eigen_vec[:,sorted_index]
        eigenvector_subset = sorted_eigenvectors[:,0:k]
        X_reduced = dot(eigenvector_subset.T(),X_meaned.T()).T()
        return X_reduced
        
     
    