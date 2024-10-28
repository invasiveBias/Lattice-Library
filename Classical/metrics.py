import sys
sys.path.append('C:/Users/LENOVO/Desktop/Lattice library/Lattice')
import Lattice_mathcomp as lt
import main
from Lattice_type import *
import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = main.as_array(y_true) if f"{type(y_true)}" != 'lattice.array' else y_true
    y_pred = main.as_array(y_pred) if f"{type(y_true)}" != 'lattice.array' else y_pred
    error = lt.nanmean((y_true - y_pred)**2)
    return error

def mean_absolute_error(y_true, y_pred):
    y_true = main.as_array(y_true) if f"{type(y_true)}" != 'lattice.array' else y_true
    y_pred = main.as_array(y_pred) if f"{type(y_true)}" != 'lattice.array' else y_pred
    error = lt.nanmean(lt.abs((y_true - y_pred)))
    return error  

def accuracy(y_true, y_pred):
    y_true = main.as_array(y_true) if f"{type(y_true)}" != 'lattice.array' else y_true
    y_pred = main.as_array(y_pred) if f"{type(y_true)}" != 'lattice.array' else y_pred
    acc = lt.nansum(y_true == y_pred) / len(y_true)
    return acc