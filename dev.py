import pandas as pd
import numpy as np
import scipy as sp
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations


class CLFRates:
    def __init__(self,
                 y, 
                 y_,
                 round=2):
        # Doing a crosstab to save some time
        self.tab = pd.crosstab(y_, y)
        
        # Getting the basic counts
        tn = self.tab.iloc[0, 0]
        fn = self.tab.iloc[0, 1]
        fp = self.tab.iloc[1, 0]
        tp = self.tab.iloc[1, 1]
        
        # Calculating the rates
        self.tnr = np.round(tn / (tn + fp), round)
        self.tpr = np.round(tp / (tp + fn), round)
        self.fnr = np.round(fn / (fn + tp), round)
        self.fpr = np.round(fp / (fp + tn), round)
        self.acc = (tn + tp) / len(y)
        
        return


def roc_coords(y, y_, round=4):
    # Getting hte counts
    tab = pd.crosstab(y_, y)
    tn = tab.iloc[0, 0]
    fn = tab.iloc[0, 1]
    fp = tab.iloc[1, 0]
    tp = tab.iloc[1, 1]
    
    # Calculating the rates
    tpr = np.round(tp / (tp + fn), round)
    fpr = np.round(fp / (fp + tn), round)
    
    return (fpr, tpr)


def pred_from_pya(y_, a, pya):
    groups = np.unique(a)
    out = np.zeros(y_.shape[0])
    for i, g in enumerate(groups):
        p = pya[i]
        pos = np.where((a == g) & (y_ == 1))[0]
        neg = np.where((a ==  g) & (y_ == 0))[0]
        out[neg] = np.random.binomial(1, p[0], len(neg))
        out[pos] = np.random.binomial(1, p[1], len(pos))
    
    return out.astype(np.uint8)
    

class PredictionBalancer:
    def __init__(self, 
                 y, 
                 y_, 
                 a,
                 round=4):
        
        self.y = y
        self.y_ = y_
        self.a = a
        self.groups = np.unique(a)
        
        # Getting the row numbers for each group
        group_ids = [np.where(a == g)[0] for g in self.groups]
        
        # Getting the proportion for each group
        self.p = [np.round(len(cols) / len(y), round) for cols in group_ids]
        
        # Calcuating the groupwise classification rates
        group_rates = [CLFRates(y[i], y_[i]) for i in group_ids]
        self.group_rates = dict(zip(self.groups, group_rates))
        self.overall_rates = CLFRates(y, y_)
    
    def fit(self):
        # Pulling things from the class
        s = self.overall_rates.acc
        e = 1 - s
        p = self.p
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(s - e) * prop, (e - s) * prop]
                             for prop in p]).flatten()
        obj_bounds = [(0, 1)]
        
        # Generating the pairs for comparison
        n_groups = len(self.groups)
        group_combos = list(combinations(self.groups, 2))
        id_combos = list(combinations(range(n_groups), 2))
        
        # Pair drop to keep things full-rank with 3 or more groups
        if n_groups > 2:
            del group_combos[-1]
            del id_combos[-1]
        
        col_combos = np.array(id_combos) * 2
        n_pairs = len(group_combos)
        
        # Making empty matrices to hold the pairwise constraint coefficients
        tprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        fprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        
        # Filling in the constraint matrices
        for i, cols in enumerate(col_combos):
            # Fetching the group-specific rates
            gc = group_combos[i]
            g0 = self.group_rates[gc[0]]
            g1 = self.group_rates[gc[1]]
            
            # Filling in the group-specific coefficients
            tprs[i, cols[0]] = g0.fnr
            tprs[i, cols[0] + 1] = g0.tpr
            tprs[i, cols[1]] = -g1.fnr
            tprs[i, cols[1] + 1] = -g1.tpr
            
            fprs[i, cols[0]] = g0.tnr
            fprs[i, cols[0] + 1] = g0.fpr
            fprs[i, cols[1]] = -g1.tnr
            fprs[i, cols[1] + 1] = -g1.fpr
        
        roc_coefs = np.vstack((tprs, fprs))
        self.roc_coefs = roc_coefs
        roc_bounds = np.zeros(roc_coefs.shape[0])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=roc_coefs,
                                       b_eq=roc_bounds)
        
        return
    
    def predict(self):
        pya = self.opt.x.reshape(len(self.groups), 2)
        adj = pred_from_pya(self.y_, self.a, pya)
        return adj
    
    def plot(self):
        pass
        


class ROCBalancer:
    def __init__(self):
        pass


# Importing some test data
records = pd.read_csv('records.csv')

# Keeping only the 3 most common race groups for now
records = records[(records.race == 'Black') |
                  (records.race == 'White') |
                  (records.race == 'Asian')]

# Setting the variables for the joint distribution
pcr = records.pcr_pos.values
taste = records.tastesmell_combo.values
race_bin = np.array(records.race == 'White', dtype=np.uint8)
race = records.race.values

# Testing the balancer
x = PredictionBalancer(pcr, taste, race)
x.fit()
y_adj = x.predict()

