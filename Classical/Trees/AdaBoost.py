import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
grandparent_directory = current_file.parent.parent.parent
sys.path.append(str(parent_directory))
sys.path.append(str(grandparent_directory))
from main import *
import Lattice_mathcomp as lt

out = 'out'

class Decision_stump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = ones(n_samples)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions

class Ada_Boost:
    def __init__(self, n_clf = 5):
        self.n_clf = n_clf
        
    def fit(self, X,y):
        X = array(X)
        y = array(y)
        n_samples , n_features = X.shape
        
        w = lt.full(n_samples,(1/n_samples))
        self.clfs = []
        
        for _ in range(self.n_clf):
            clf = Decision_stump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:,feature_i]
                thresholds = lt.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = ones(n_samples)
                    predictions[X_column < threshold] = -1
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error: 
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
                        
            EPS = 1e-10
            clf.alpha = 0.5 * lt.log((1.0 - min_error + EPS) / (min_error + EPS))
        
            predictions = clf.predict(X)
            
        
 
            w *= lt.exp(-clf.alpha * y * predictions)

            w /= lt.sum(w)
        
            self.clfs.append(clf)
            
    def predict(self,X):
        predictions = concatenate([clf.alpha * clf.predict(X) for clf in self.clfs],axis=1)
        predictions = lt.sign(lt.sum(predictions,dim=0))
        predictions = lt.where(predictions == 1, predictions, 0)
        return predictions