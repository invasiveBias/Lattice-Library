import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
grandparent_directory = current_file.parent.parent.parent
sys.path.append(str(parent_directory))
sys.path.append(str(grandparent_directory))
from Trees.DecisonTree import Decision_Tree
from Classical.metrics import mean_squared_error
from main import *
import Lattice_mathcomp as lt
from collections import Counter



def sigmoid(p):
    return 1 / (1 + lt.exp(-p))

def softmax(raw_predictions):
    numerator = lt.exp(raw_predictions) 
    denominator = lt.sum(lt.exp(raw_predictions), dim=1).reshape(-1, 1)
    return numerator / denominator

def converter(v,lst):
    n_lst = []
    for i in lst:
        if i == v:
            n_lst.append(1)
        else:
            n_lst.append(0)
    return array(n_lst)

        

class Gradient_Boost:
    def __init__(self, n_learners= 5, max_depth=5,gamma = 0.1,min_sample_split= 2, n_feats=None, mode='cls',mclass =False):
        self.n_learners = n_learners
        self.gamma = gamma
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.mode = 'grd' if mode == 'cls' else mode
        self.is_mclass = mclass
        
        
    def fit(self, X,y):
        self.X = array(X)
        y = array(y)
        n_samples , n_features = self.X.shape
        
        def training_sequence(vt: tuple):
            y, base_pred = vt
            built_trees = []
            for _ in range(self.n_learners):
                if self.mode == 'rgs':
                    y_ = y - base_pred
                    tree = Decision_Tree(min_sample_split=self.min_sample_split, max_depth=self.max_depth,n_feats=self.n_feats, mode=self.mode)
                    tree.fit(self.X,y_)
                    predictions = tree.predict(self.X)
                    base_pred = base_pred + (self.gamma * predictions)
                    built_trees.append(tree)
        
                else:
                    prob = sigmoid(base_pred)
                    y_ = y - prob
                    tree = Decision_Tree(min_sample_split=self.min_sample_split, max_depth=self.max_depth,n_feats=self.n_feats, mode=self.mode)
                    tree.fit(self.X,y_)
                    predictions = tree.predict(self.X)
                    grad_output = array([lt.sum(pred)/ ((prob_i * (1 - prob_i)) * len(pred)) for pred, prob_i in zip(predictions,prob)])
                    base_pred = prob + self.gamma * grad_output
                    built_trees.append(tree)
                    
            return built_trees
            
        
        if self.mode == 'rgs':
            base_pred = lt.full(n_samples,lt.mean(y))
            self.leaf = base_pred
            self.trees = training_sequence((y, base_pred))
        else:
            self.classes = lt.unique(y)
            if len(self.classes) <= 2:
                prob_pst = lt.sum(y[y== 1])/ lt.sum(lt.where(y > 0,y, y + 1))
                log_odds = lt.log(prob_pst/(1-prob_pst))
                log_odds = lt.full(n_samples,log_odds)
                self.leaf = sigmoid(log_odds)
                self.trees = training_sequence((y, log_odds))
            else:
                y_ohe = lt.one_hot(y)
                self.classes = lt.unique(y)
                prob_pst = [lt.sum(y_ohe[y_ohe[:,c]== 1])/ lt.sum(lt.where(y_ohe[:,c] > 0,y_ohe[:,c], y_ohe[:,c] + 1)) for c in self.classes]
                base_pred = lt.tile(prob_pst,[n_samples,1])
                self.leaf = lt.tile(prob_pst,[n_samples,1])
                self.forest = []
                for m in range(self.n_learners):
                    prob = sigmoid(base_pred)
                    y_hat = softmax(prob)
                    residual = y_ohe - y_hat
                    tree_cluster = []
                    for c in self.classes:
                        tree = Decision_Tree(min_sample_split=self.min_sample_split, max_depth=self.max_depth,n_feats=self.n_feats, mode=self.mode)
                        tree.fit(X,residual[:,c])
                        prediction = tree.predict(X)
                        tree_output = array([lt.sum(pred)/ ((prob_i * (1 - prob_i)) * len(pred)) for pred, prob_i in zip(prediction,prob[:,c])])
                        base_pred[:,c] +=  self.gamma * tree_output
                        tree_cluster.append(tree)
                    self.forest.append(tree_cluster)
                

                
    
    
    def predict(self, X):
        X = array(X)
        if self.mode == 'rgs':
            n_samples = X.shape[0]
            tree_preds = array([tree.predict(X) * self.gamma for tree in self.trees])
            pred = array([lt.sum(y) for y in tree_preds.T()])
            pred = self.leaf[n_samples] + pred
            return pred
        else:
            def calculate_value(leaf, trees, X,mc= False):
                grad_output = 0
                for tree in trees:
                    predictions = tree.predict(X)
                    grad_output = [lt.sum(pred)/ ((prob_i * (1 - prob_i)) * len(pred)) for pred, prob_i in zip(predictions,leaf)]
                    grad_output += self.gamma * array(grad_output)
                log_odds = leaf + grad_output
                if mc:
                    return log_odds
                pred = sigmoid(log_odds)
                arr = []
                for p in pred:
                    if p < 0.5:
                        arr.append(0)
                    else:
                        arr.append(1)
                return arr
            
            n_samples = X.shape[0]
            if self.is_mclass == False:
                leaf = self.leaf[:n_samples]
                return array(calculate_value(leaf,self.trees,X))
            else:
                leaf = self.leaf[:n_samples]
                for cluster in self.forest:
                    for c in self.classes:
                        prediction = cluster[c].predict(X)
                        tree_output = array([lt.sum(pred)/ ((prob_i * (1 - prob_i)) * len(pred)) for pred, prob_i in zip(prediction,leaf[:,c])])
                        leaf[:,c] += self.gamma * tree_output

                
                

                    

        
                
                
                
                
            
                
     
                