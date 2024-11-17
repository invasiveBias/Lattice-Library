import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))
import main
import Lattice_mathcomp as lt
import numpy as np




class LinearRegression:
    def __init__(self, lr=0.0001, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    
    def fit(self, X,y):
        X = main.as_array(X)
        y = main.as_array(y)
        n_samples, n_features = X.shape
        self.weights = main.zeros(n_features)
        self.bias = main.zeros(1)
        print(f"Beginning Training: fitting data of shape {X.shape} to Labels")
        for _ in range(self.n_iters):
            y_pred = lt.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * lt.dot(X.T(),(y_pred - y))
            db = (1/n_samples) * lt.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        print('Training complete!!!')
    def predict(self, X):
        X = main.as_array(X)
        y_pred = lt.dot(X,self.weights) + self.bias
        return y_pred
            