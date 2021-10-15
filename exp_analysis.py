import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

import tools
import balancers

# Reading in some datasets
compas = pd.read_csv('~/Downloads/archive/cox-violent-parsed.csv')
tb = pd.read_csv('~/code/ml-workshop/data/tb.csv')
exp = pd.read_csv('~/Desktop/exp_stats.csv')

'''Part 1: Working with COMPAS'''
# Building an RF to predict recidivism
compas = compas[compas.is_recid != -1].reset_index(drop=True)
score = compas.decile_score
score_cut = pd.qcut(score, [0, .2, .8, 1],
                    labels=['Low', 'Medium', 'High'])
cat_cols = ['sex', 'race']
num_cols = ['age', 'juv_fel_count', 'juv_misd_count',
            'priors_count']
cat_sparse = pd.concat([tools.sparsify(compas[c],
                                       long_names=True)
                        for c in cat_cols], axis=1)
X = pd.concat([cat_sparse, 
               compas[num_cols]], 
              axis=1)
y = compas.is_recid
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_[:, 1]
prob_cut = pd.qcut(probs, [0, .2, .8, 1],
                   labels=['Low', 'Medium', 'High'])

# Checking basic metrics like Brier score
rf_brier = tools.brier_score()
