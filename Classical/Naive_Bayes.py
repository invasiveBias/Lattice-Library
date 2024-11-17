import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))
import Lattice_mathcomp as lt
from Lattice_type import array
import numpy as np
from main import *


class NaiveBayes():
    
    def fit(self, X,y):
        X = array(X)
        y = array(y)
        n_samples, n_features = X.shape
        self.classes = lt.unique(y)
        n_classes = len(self.classes)
    
        self.mean = zeros((n_classes,n_features),dtype=float)
        self.var = zeros((n_classes,n_features),dtype=float)
        self.priors = zeros((n_classes),dtype=float)
    
        for c in self.classes:
            X_c =  X[c==y]
            self.mean[c,:] = X_c.mean(dim=0)
            self.var[c,:] = X_c.var(dim=0)
            self.priors[c] = X_c.shape[0] / float(n_samples)

            
    def predict(self,X):
         X = array(X)
         y_pred = [self._predict(x) for x in X]
         return concatenate(y_pred)
                
    def _predict(self,x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = lt.log(self.priors[idx])
            class_conditional = lt.sum(lt.log(self.pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
            
        posteriors = concatenate(posteriors)
        return self.classes[lt.argmax(posteriors)]
        
    
    def pdf(self,class_idx,x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = lt.exp(- (x-mean) **2 / (2 * var))
        pi = 3.14
        denominator = lt.sqrt(2 * pi * var)
        return numerator / denominator
        
        