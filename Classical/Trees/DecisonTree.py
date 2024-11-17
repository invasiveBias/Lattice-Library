import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
grandparent_directory = current_file.parent.parent.parent
sys.path.append(str(parent_directory))
sys.path.append(str(grandparent_directory))
from main import *
import Lattice_mathcomp as lt
from collections import Counter


def entropy(y):
    y = array(y).astype(int) if type(y) != array else y.astype(int)
    hist = lt.bincount(y)
    ps = hist/len(y)
    return -lt.sum([p * lt.log2(p) for p in ps if p > 0])

def mse(y):
    mean = lt.mean(y)
    return lt.mean((y - mean)**2)


class Node:
    def __init__(self, feature=None, threshold= None, left=None, right=None,*,value= None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class Decision_Tree:
    def __init__(self, min_sample_split= 2, max_depth = 30, n_feats=None, mode='cls'):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.mode = mode
        if self.mode not in ['cls','rgs','grd']:
            raise Exception(f'Mode can only be "cls" for classification, "rgs" for regression or "grd" for gradient boosting methods and not {mode}')
    
    def fit(self,X,y):
        X = array(X) if type(X) != array else X
        y = array(y) if type(y) != array else y
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self.grow_tree(X,y)
    
    def grow_tree(self,X,y, depth=0,default_value=0):
        n_samples, n_features = X.shape
        n_labels = len(lt.unique(y))

        
        if depth>= self.max_depth or n_labels <= 1 or  n_samples < self.min_sample_split:
            if len(y) == 0:
                leaf_value = default_value if default_value is not None else 0
            else:
                if self.mode == 'cls':
                    leaf_value = self.most_common_label(y)  
                elif self.mode == 'rgs':
                    leaf_value = self.average_value(y)
                else:
                    leaf_value = y
            return Node(value=leaf_value)
        
        feat_idxs = choice(n_features,self.n_feats, replace= False)
        
        best_feat, best_thresh = self.best_criteria(X,y,feat_idxs, entropy) if self.mode == 'cls' else self.best_criteria(X,y,feat_idxs, mse)
        left_idxs, right_idxs = self.split(X[:,best_feat], best_thresh)
        left = self.grow_tree(X[left_idxs,:], y[left_idxs],depth=depth+1,default_value=default_value)
        default_value = left.value
        right = self.grow_tree(X[right_idxs,:], y[right_idxs],depth=depth+1,default_value=default_value)
        default_value = right.value
        return Node(best_feat,best_thresh,left,right)
        
    
    
    def best_criteria(self,X,y, feat_idxs,split_func):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            x_column = X[:,feat_idx]
            thresholds = lt.unique(x_column)
            for threshold in thresholds:
                gain = self.information_gain(y, x_column,threshold,split_func)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def information_gain(self,y, x_column, split_thresh,split_func):
        parent_randomness = split_func(y)
        left_idxs, right_idxs = self.split(x_column,split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        nl, nr = len(left_idxs), len(right_idxs)
        rand_l, rand_r = split_func(y[left_idxs]), split_func(y[right_idxs])
        child_randomness = (nl/n) * rand_l + (nr/n) * rand_r
        ig = parent_randomness - child_randomness
        return ig
        
    def split(self,x_column,split_thresh):
        left_idx = lt.argwhere(x_column <= split_thresh).flatten()
        right_idx = lt.argwhere(x_column > split_thresh).flatten()
        return left_idx, right_idx
    
    def most_common_label(self,y):
        counter = Counter(y.tolist())
        most_comon = counter.most_common(1)[0][0]
        return most_comon
    
    def average_value(self,y):
        return lt.mean(y)
    
    def predict(self,X):
        X = array(X)
        if self.mode == 'cls' or self.mode == 'rgs':
            return array([self.traverse_tree(x,self.root) for x in X])
        else:
            return [self.traverse_tree(x,self.root) for x in X]
    

    def traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
        
