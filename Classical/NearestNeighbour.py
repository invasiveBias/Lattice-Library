import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))
import numpy as np
from collections import Counter
import Lattice_mathcomp as lt
from Lattice_type import array
from main import concatenate


def Euclidean(x1,x2):
    x1 = array(x1)
    x2 = array(x2)
    return lt.sqrt(lt.sum((x1 - x2)**2))
    
class KNN:
    def __init__(self, k,mode='cls'):
        self.k = k
        self.mode = mode
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        predicted_labels = []
        for x in X:
            distances = concatenate([Euclidean(x,x_train) for x_train in self.X_train])
            k_indices = lt.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            if self.mode == 'cls':
                most_common = Counter(k_nearest_labels).most_common(1)
                predicted_labels.append(most_common[0][0])
            elif self.mode == 'rgs':
                predicted_labels.append(lt.mean(k_nearest_labels))
            else:
                return f"modes can either be 'cls' for classification or 'rgs' for regression, not {self.mode}"
        return array(predicted_labels)
    
