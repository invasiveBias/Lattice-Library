import sys
sys.path.append('C:/Users/LENOVO/Desktop/Lattice library/Lattice')
sys.path.append('C:/Users/LENOVO/Desktop/Lattice library/Lattice/Classical/Trees')
from DecisonTree import Decision_Tree
from main import *
import Lattice_mathcomp as lt
from collections import Counter

def bootstrap_sample(X,y):
    X = array(X)
    y = array(y)
    n_samples, n_features = X.shape
    idxs = lt.choice(n_samples, size=n_samples, replace=True)
    sub_set = X[idxs], y[idxs]
    return sub_set

def most_common_label(y):
    counter = Counter(y.tolist())
    most_comon = counter.most_common(1)[0][0]
    return most_comon

class Random_Forest():
    def __init__(self, n_trees=10,min_sample_split= 2, max_depth = 30, n_feats=None, mode='cls'):
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.mode = mode
        self.trees = []
        
    def fit(self, X,y):
        for _ in range(self.n_trees):
            tree = Decision_Tree(min_sample_split=self.min_sample_split, max_depth=self.max_depth,n_feats=self.n_feats, mode=self.mode)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        if self.mode == 'cls':
            tree_preds = array([tree.predict(X) for tree in self.trees])
            pred = [most_common_label(y) for y in tree_preds.T()]
            return array(pred)
        if self.mode == 'rgs':
            tree_preds = array([tree.predict(X) for tree in self.trees])
            pred = [lt.mean(y) for y in tree_preds.T()]
            return array(pred)