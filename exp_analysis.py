import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

import tools
import balancers

# Reading in the experimental data
exp = pd.read_csv('~/Desktop/exp_stats.csv')

'''Part 1: Working with COMPAS'''
# Building an RF to predict recidivism
compas = pd.read_csv('~/Downloads/archive/cox-violent-parsed.csv')

compas = compas[(compas.is_violent_recid != -1) &
                (compas.v_decile_score != -1)]
compas = compas[(compas.race == 'Caucasian') |
                (compas.race == 'African-American')]
compas = compas.reset_index(drop=True)

race = compas.race.values
score = compas.v_decile_score
score_cut = pd.cut(score, [0, 2, 8, 10],
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True).values
cat_cols = ['sex', 'race']
num_cols = ['age', 'juv_fel_count', 'juv_misd_count',
            'priors_count']
cat_sparse = pd.concat([tools.sparsify(compas[c],
                                       long_names=True)
                        for c in cat_cols], axis=1)
X = pd.concat([cat_sparse, 
               compas[num_cols]], 
              axis=1)
y = compas.is_violent_recid
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_[:, 1]
prob_cut = pd.cut(probs, [0, .2, .8, 1],
                  labels=['Low', 'Medium', 'High'],
                  include_lowest=True)

# Checking some basic classification metrics
rf_stats = tools.clf_metrics(y, probs)
cp_stats = tools.clf_metrics(y, score/10)

# And balancing the cp scores against the RF
b = balancers.MulticlassBalancer(np.array([p for p in prob_cut]), 
                                 np.array([s for s in score_cut]), 
                                 race)
b.adjust(goal='odds', loss='macro')
b.summary()
b.plot()

'''Part 2: Working with the TB data'''
# Reading in the data
tb = pd.read_csv('~/code/ml-workshop/data/tb.csv')
tb_bin = pd.read_csv('~/code/ml-workshop/data/tb_bin.csv')

# Filtering out folks with missing observations
no_na = np.where((np.sum(np.isnan(tb_bin.values), axis=1) == 0) &
                 ([not np.isnan(c) for c in tb.cd4_cnt]))[0]
sex = tb_bin.sex.values.astype(str)[no_na]
cd4 = tb.cd4_cnt.values[no_na]
cd4_levels = ['Low', 'Medium', 'High']
cd4_cut = pd.cut(cd4,
                 [0, 200, 500, 2000],
                 include_lowest=True,
                 labels=cd4_levels)
    
tb_y = np.array([p for p in cd4_cut])
tb_X = tb_bin.drop(['sex'], axis=1).values[no_na]

tb_rf = RandomForestClassifier(n_estimators=1000,
                               n_jobs=-1,
                               oob_score=True)
tb_rf.fit(tb_X, tb_y)
tb_probs = tb_rf.oob_decision_function_
tb_preds = np.array([cd4_levels[i] for i in np.argmax(tb_probs, axis=1)])

b = balancers.MulticlassBalancer(tb_y, tb_preds, sex)
