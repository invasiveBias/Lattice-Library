import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))
from main import *
import Lattice_mathcomp as lt


# uses a one vs rest approach to create hyperplanes when classes >= 3
class SVC:
    def __init__(self,learning_rate=0.001, lambda_param = 0.01, n_iters= 1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.multi_class = None
        self.lb_dict = {}
    
    
    def fit(self,X,y):
        X = array(X)
        y = array(y)
        classes = lt.unique(y)
        if len(classes) > 2:
            self.multi_class = True
            for lbl in classes:
                self.lb_dict.update({lbl.item() : self.operation(X,y, y != lbl,mtc=True)})
        else:
            self.operation(X,y,y <=0)
    def operation(self,X,y,condition,mtc= False):        
        y_ = lt.where(condition, -1,1)
        n_samples, n_features = X.shape
        self.weights = zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (lt.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - lt.dot(x_i,y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
        if mtc:
            return [self.weights,self.bias]
                        
    def predict(self,x):
        x = array(x)
        if self.multi_class:
            scores = []
            for k,v in self.lb_dict.items():
                linear_output = lt.dot(x,v[0]) - v[1]
                scores.append(linear_output)
            scores = array(scores).reshape(len(x),len(self.lb_dict.keys()))
            return lt.argmax(scores,axis=1)
            
        else:
            linear_output = lt.sign(lt.dot(x, self.weights) - self.bias)
            linear_output = [0 if ln == -1 else 1 for ln in linear_output]
            return linear_output
            
                    


class SVR:
    def __init__(self,learning_rate=0.001, lambda_param = 0.01, n_iters= 1000,i_loss=0.1):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.i_loss = i_loss
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        X = array(X)
        y = array(y)
        n_samples, n_features = X.shape
        self.weights = zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                ub_loss = y[idx] - (lt.dot(x_i, self.weights) - self.bias)
                lb_loss = (lt.dot(x_i, self.weights) - self.bias) - y[idx]
                condition = ub_loss <= self.i_loss or lb_loss >= -self.i_loss
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)  - x_i
                    self.bias -= self.learning_rate 
     
     
                    
    def predict(self,x):
        x = array(x)
        linear_output = lt.dot(x, self.weights) - self.bias
        return linear_output