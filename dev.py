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
    
    def fit(y, probs, a, return_optima=True):
        self.y = y
        self.probs = probs
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


# Importing some test data
records = pd.read_csv('records.csv')

# Keeping only the 3 most common race groups for now
records = records[(records.race == 'Black') |
                  (records.race == 'White') |
                  (records.race == 'Asian')]

# Setting the variables for the joint distribution
pcr = np.repeat(records.pcr_pos.values, 10)
cough = np.repeat(records.cough.values, 10)
fever = np.repeat(records.fever_chills.values, 10)
taste = np.repeat(records.tastesmell_combo.values, 10)
race_bin = np.repeat(np.array(records.race == 'White', 
                              dtype=np.uint8), 10)
race = np.repeat(records.race.values, 10)

# Testing the balancer
pb = b.PredictionBalancer()
pb.fit(pcr, taste, race)
pb.summary()
pb.plot()

