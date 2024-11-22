import numpy as np
from numpy.random import rand, randn,  randint
from sklearn.decomposition import PCA
from main import *
from Lattice_type import *



# vector transformations

def sum(val,dim= None):
    val = array(val) if type(val) != array else val
    arr = np.sum(val._data,axis=dim) 
    return array(arr,requires_grad = val.requires_grad)
    
    
def mean(val,dim=None):
    val = array(val) if type(val) != array else val
    arr = np.mean(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def median(val,dim=None):
    val = array(val) if type(val) != array else val
    arr = np.median(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def nanmean(val,dim=None):
    val = array(val) if type(val) != array else val
    arr = np.nanmean(val._data,dim)
    return array(arr, requires_grad = val.requires_grad)

def nansum(val,dim= None):
    val = array(val) if type(val) != array else val
    arr = np.nansum(val._data,axis=dim) 
    return array(arr,requires_grad = val.requires_grad)

def norm(val):
    val = array(val) if type(val) != array else val
    arr = np.linalg.norm(val._data)
    return array(arr, requires_grad = val.requires_grad)

def dot(val,val2):
        arr = np.dot(val._data,val2._data)
        requires_grad = val.requires_grad or val2.requires_grad
        return array(arr, requires_grad = requires_grad)
    
def unique(val,return_counts= False):
    val = array(val) if type(val) != array else val
    arr = np.unique(val._data,return_counts= return_counts) 
    if return_counts == False:
        return array(arr, requires_grad = val.requires_grad)
    else:
        return array(arr[0], requires_grad = val.requires_grad) , arr[1]


# element-wise transformations

def abs(val,*args):
    val = array(val) if type(val) != array else val
    arr = np.abs(val._data,*args)
    return array(arr)

def exp(val):
    val = array(val) if type(val) != array else val
    arr = np.exp(val._data.astype(float))
    return array(arr)

def log(val):
    val = array(val) if type(val) != array else val
    arr = np.log(val._data.astype(float))
    return array(arr)

def log10(val):
    val = array(val) if type(val) != array else val
    arr = np.log10(val._data.astype(float))
    return array(arr)

def log2(val):
    val = array(val) if type(val) != array else val
    arr = np.log2(val._data.astype(float))
    return array(arr)


def sqrt(val):
    val = array(val) if type(val) != array else val
    arr = np.sqrt(val._data.astype(float))
    return array(arr)

def argsort(val,*args, **kwargs):
    val = array(val) if type(val) != array else val
    ind = np.argsort(val._data)
    return array(ind)

def argmax(val,axis=None,*args, **kwargs):
    val = array(val) if type(val) != array else val
    ind = np.argmax(val._data,axis=axis)
    return array(ind)

def fill_diagonal(val1, val2):
    val1 = array(val1) if type(val1) != array else val1
    val2 = array(val2) if type(val2) != array else val2
    np.fill_diagonal(val1._data,val2._data)
    return val1

def cov(val, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
    val = array(val) if type(val) != array else val
    arr = np.cov(val._data, y=y, rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights, aweights=aweights, dtype=dtype)
    return array(arr)
    
def where(condition, x, y):
    val = np.where(condition._data, x,y)
    return array(val)

def sign(val):
    val = array(val) if type(val) != array else val
    val = np.sign(val._data)
    return array(val)

def bincount(val):
    val = array(val) if type(val) != array else val
    arr = np.bincount(val._data)
    return array(arr) 

def argwhere(val):
    val = array(val) if type(val) != array else val
    arr = np.argwhere(val._data)
    return array(arr)

def percentile(val,q,axis=None):
    val = array(val) if type(val) != array else val
    arr = np.percentile(val._data,q,axis=axis)
    return array(arr)

def full(shape, fill_value, dtype=None):
    fill_value = array(fill_value) if type(fill_value) != array else fill_value
    arr = np.full(shape=shape, fill_value=fill_value._data,dtype=dtype)
    return array(arr)

def one_hot(arr):
    arr = array(arr) if type(arr) != array else arr
    classes = unique(arr)
    def converter(v,lst):
        n_lst = []
        for i in lst:
            if i == v:
                n_lst.append(1)
            else:
                n_lst.append(0)
        return n_lst
    ohm = empty((arr.shape[0], len(classes)))
    for v in classes:
        ohm[:,v] = converter(v,arr)
    return ohm

def tile(A, reps):
    A = np.array(A) if type(A) != array else A._data
    return array(np.tile(A, reps))


class matrix_comp():
    def __init__(Self):
        pass
       
    @staticmethod
    def matmul(val1,val2):
        val1 = array(val1) if type(val1) != array else val1
        val2 = array(val2) if type(val2) != array else val2
        obj = val1 @ val2
        return obj
    
    
    @staticmethod
    def mat_inv(val):
        val = array(val) if type(val) != array else val
        val = np.linalg.inv(val._data)
        return array(val)
    
    @staticmethod
    def eigen(val):
        val = array(val) if type(val) != array else val
        e_val, e_vec = np.linalg.eig(val._data)
        return array(e_val.real), array(e_vec.real)
    
    @staticmethod
    def det(val):
        val = array(val) if type(val) != array else val
        val = np.linalg.det(val._data)
        return array(val)
    
    @staticmethod
    def svd(val,herm=False):
        val = array(val) if type(val) != array else val
        u,d,vt = np.linalg.svd(val._data,hermitian=herm)
        return array(u), array(d), array(vt)
    
    @staticmethod
    def pinv(val):
        val = array(val) if type(val) != array else val
        val = np.linalg.pinv(val._data)
        return array(val)
    
    @staticmethod
    def trace(val):
        val = array(val) if type(val) != array else val
        arr = np.trace(val._data)
        return array(arr)
    
    
    @staticmethod
    def pca(X,k=2):
        X = array(X) if type(X) != array else X
        X_meaned = X - mean(X , dim = 0)
        cov_mat = cov(X_meaned , rowvar = False)
        eigen_val, eigen_vec = matrix_comp.eigen(cov_mat)
        sorted_index = argsort(eigen_val)[::-1]
        sorted_eigenvalue = eigen_val[sorted_index]
        sorted_eigenvectors = eigen_vec[:,sorted_index]
        eigenvector_subset = sorted_eigenvectors[:,0:k]
        X_reduced = dot(eigenvector_subset.T(),X_meaned.T()).T()
        return X_reduced
        
     
    