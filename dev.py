import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from importlib import reload
from matplotlib import pyplot as plt
from itertools import combinations

import balancers as b
from tools import group_roc_coords, pred_from_pya, CLFRates


class ProbabilityBalancer:
    def __init__(self):
        pass
    
    def fit(self, 
            y, 
            probs, 
            a,
            round=4, 
            return_optima=True):
        
        self.y = y
        self.probs = probs
        self.a = a
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]
        
        # Getting the proportion for each group
        self.p = [(np.sum(a == g) / len(y)) for g in self.groups]
        
        # Getting the raw ROC info
        rocs = [roc_curve(y[ids], probs[ids]) for ids in group_ids]
        self.rocs = rocs

# Testin gthe balancer
pb = ProbabilityBalancer()
pb.fit(pcr, probs, race_bin)


# Setting the variables for the joint distribution
pcr = records.pcr.values
cough = records.cough.values
fever = records.fever.values
taste = records.losstastesmell.values
race_bin = np.array(records.race == 'White', dtype=np.uint8)
race = records.race.values
X = records.iloc[:, 3:18].values

# Fitting a toy model
rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X, pcr)
probs = rf.oob_decision_function_[:, 1]

# Testin gthe balancer
pb = ProbabilityBalancer()
pb.fit(pcr, probs, race)

# Importing some test data
records = pd.read_csv('records.csv')

# Keeping only the 3 most common race groups for now
records = records[(records.race == 'Black / African American') |
                  (records.race == 'White') |
                  (records.race == 'Asian') |
                  (records.race == 'Undisclosed')]
