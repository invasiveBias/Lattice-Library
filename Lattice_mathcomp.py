import numpy as np
from numpy.random import rand, randn,  randint
from sklearn.decomposition import PCA
from main import *
from Lattice_type import *



def obj_alloc(num):
    if 'int' in f'{type(num)}':
        return Integer(num)
    elif 'float' in f'{type(num)}':
        return Float(num)
    else:
        return array(num)


class matrix_comp():
    def __init__(Self):
        pass
    
    @staticmethod
    def norm(val):
        obj = to_numpy(val._data)
        obj = np.linalg.norm(obj)
        return obj
    
    @staticmethod
    def matmul(val1,val2):
        obj = val1 @ val2
        return obj
    
    @staticmethod
    def mat_inv(val):
        obj = to_numpy(val._data)
        obj = np.linalg.inv(obj)
        return obj_alloc(obj)
    
    @staticmethod
    def eigen(val):
        obj = to_numpy(val)
        e_val, e_vec = np.linalg.eig(obj)
        return obj_alloc(e_val.real), obj_alloc(e_vec.real)
    
    @staticmethod
    def det(val):
        obj = to_numpy(val)
        det_ = np.linalg.det(obj)
        return obj_alloc(det_)
    
    @staticmethod
    def svd(val,herm=False):
        obj = to_numpy(val)
        u,d,vt = np.linalg.svd(obj,hermitian=herm)
        return obj_alloc(u), obj_alloc(d), obj_alloc(vt)
    
    @staticmethod
    def pinv(val):
        obj = to_numpy(val)
        inval = np.linalg.pinv(obj)
        return obj_alloc(inval)
    
    @staticmethod
    def trace(val):
        obj = to_numpy(val)
        tra = np.trace(obj)
        return obj_alloc(tra)
    
    
    @staticmethod
    def pca(X,k=2):
        X_meaned = X - np.mean(X , axis = 0)
        cov_mat = np.cov(X_meaned , rowvar = False)
        eigen_val, eigen_vec = np.linalg.eig(cov_mat)
        sorted_index = np.argsort(eigen_val)[::-1]
        sorted_eigenvalue = eigen_val[sorted_index]
        sorted_eigenvectors = eigen_vec[:,sorted_index]
        eigenvector_subset = sorted_eigenvectors[:,0:k]
        X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()
        return obj_alloc(X_reduced)
        
     
     
     
class stat_comp():
    def __init__(Self):
        pass
    
    @staticmethod
    def sum(val,dim= None):
        obj = to_numpy(val)
        sum_ = np.sum(val,dim) 
        return obj_alloc(sum_)

    

