import pandas as pd
import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier


class CLFRates:
    def __init__(self,
                 y, 
                 y_,
                 round=2):
        # Doing a crosstab to save some time
        tab = pd.crosstab(y_, y)
        
        # Getting the basic counts
        tn = tab.iloc[0, 0]
        fn = tab.iloc[0, 1]
        fp = tab.iloc[1, 0]
        tp = tab.iloc[1, 1]
        
        # Calculating the rates
        tnr = np.round(tn / (tn + fp), round)
        tpr = np.round(tp / (tp + fn), round)
        fnr = np.round(fn / (fn + tp), round)
        fpr = np.round(fp / (fp + tn), round)
        
        # Setting the class attributes
        self.tnr = tnr
        self.tpr = tpr
        self.fnr = fnr
        self.fpr = fpr
        self.tn = tn
        self.fn = fn
        self.fp = fp
        self.tp = tp
        self.tab = tab

# Importing some test data
records = pd.read_csv('records.csv')

# Setting the variables for the joint distribution
pcr = records.pcr_pos.values
taste = records.tastesmell_combo
race = np.array(records.race == 'White', dtype=np.uint8)


def from_top_left(x, y, round=4):
    d = np.sqrt(x**2 + (y - 1)**2)
    return np.round(d, round)


def balance(y, y_, a):
    # Setting the group IDs and predictor stats
    a0_idx = np.where(a == 0)[0]
    a1_idx = np.where(a == 1)[0]
    a0 = CLFRates(y[a0_idx], y_[a0_idx])
    a1 = CLFRates(y[a1_idx], y_[a1_idx])
    rates = CLFRates(y, y_)
    
    # Setting up the linear program
    dpr = rates.fpr - rates.tpr
    p1 = np.sum(a) / len(a)
    p0 = 1 - p1
    obj = np.array([dpr * p0, 
                    -dpr * p0, 
                    dpr * p1, 
                    -dpr * p1])
    obj_bounds = [(0, 1)]
    tpr_b_coef = np.array([1 - a0.tpr,
                           a0.tpr,
                           1 - a1.tpr,
                           a1.tpr])
    fpr_b_coef = np.array([a0.fpr,
                           1 - a0.fpr,
                           a1.fpr,
                           1 - a1.fpr])
    A_eq = np.vstack((tpr_b_coef, fpr_b_coef))
    b_eq = np.array([0, 0])
    
    # Running the optimization
    opt = sp.optimize.linprog(c=obj,
                              bounds=obj_bounds,
                              A_eq=A_eq,
                              b_eq=b_eq)
    return A_eq, opt


