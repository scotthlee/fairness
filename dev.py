import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_curve, hinge_loss
from sklearn.ensemble import RandomForestClassifier
from importlib import reload
from matplotlib import pyplot as plt
from itertools import combinations

import balancers as b
import tools


# Reading in the data
df = pd.read_csv('farm_animals.csv')

# Making a single binary for shear
shear = np.array(df.action == 'shear', dtype=np.uint8)
shear_pred = np.array(df.pred_action == 'shear', dtype=np.uint8)

# Setting up the multiple binaries
y_bin = [np.array(y == c, dtype=np.uint8) for c in actions]
pred_bin = [np.array(y_ == c, dtype=np.uint8) for c in actions]

# Running the individual optimizations
pbs = [b.PredictionBalancer(y_bin[i], pred_bin[i], a)
       for i in range(len(y_bin))]
[bal.adjust() for bal in pbs]
bal_probs = np.array([bal.pya for bal in pbs])

def conditional_matrix(pyas):
    n_groups = pyas[0].shape[0]
    n_outcomes = len(pyas)
    out = np.zeros(shape=(n_outcomes, n_groups, n_groups))
    out_range = range(n_outcomes)
    for g in range(n_groups):
        for col in out_range:
            rows = list(out_range)
            row = rows[col]
            out[g][row][col] = pyas[g][col][1]
            del rows[row]
            for r in rows:
                out[g][r][col] = pyas[g][col][0]
    return out
        

