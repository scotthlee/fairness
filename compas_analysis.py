import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from multiprocessing import Pool
from matplotlib import pyplot as plt

import tools
import balancers


'''Part 1: Time stuff with COMPAS'''
# Importing the data and filtering according to the ProPublic analysis
compas_full = pd.read_csv('~/code/compas-analysis/cox-parsed.csv')
risk_levels = ['High', 'Medium', 'Low']
races = ['African-American', 'Caucasian', 'Hispanic']
compas = compas_full[(compas_full.end - compas_full.start >= 0) &
                ([s in risk_levels for s in compas_full.score_text]) &
                ([s in races for s in compas_full.race]) &
                (compas_full.event == 1)]
compas = compas.reset_index(drop=True)
score_text = compas.score_text.astype('category')
score_text = score_text.cat.reorder_categories(risk_levels)
gap = (compas.end - compas.start).values

# Combining gap info with recidivism info
#no_recid = np.where(compas.event == 0)[0]
#recid = np.where(compas.event == 1)[0]

# Using Otsu thresholding to pick risk levels based on time to crime
p_range = np.arange(0.01, 1, .01)
p_combos = [c for c in combinations(p_range, 2)]
good = np.where(np.diff(p_combos)>= .2)[0]
good_combos = [p_combos[i] for i in good]

with Pool() as p:
    input = [(gap, c, risk_levels) for c in good_combos]
    res = p.starmap(tools.otsu, input)
    p.close()
    p.join()

# Selecting the cutoff that yielded the lowest intraclass variance
best_cut = [p for p in good_combos[np.argmin(res)]]

# Thresholding the times to crime with the best cutoffs
gap_cut = pd.qcut(x=gap, 
                 q=[0] + best_cut + [1],
                 labels=risk_levels).to_list()
gap_cut = np.array(gap_cut)

# Making a tall version for visualization
gap_df = pd.DataFrame(zip(gap_cut, gap), 
                        columns=['otsu', 'days'])
gap_df['compas'] = score_text

# Making a few basic density plots
sns.set_style('darkgrid')
sns.set_palette('colorblind')

# First by race
fig, ax = plt.subplots(3, 2, sharey=True)
for i, r in enumerate(races):
    for j, measure in enumerate(['otsu', 'compas']):
        sns.kdeplot(x='days',
                      data=gap_df[compas.race == r],
                      hue=measure,
                      ax=ax[i, j],
                      multiple='stack')
        ax[i, j].set_ylabel(r)
  
fig.suptitle('Days to recidivism')
plt.show()

# And then overall
fig, ax = plt.subplots(1, 2, sharey=True)
for i, measure in enumerate(['otsu', 'compas']):
    sns.kdeplot(x='days',
                  data=gap_df,
                  hue=measure,
                  ax=ax[i],
                  multiple='stack')
fig.suptitle('Days to recidivism')
plt.show()

'''
# Using a random forest for balancing because compas sucks
race = compas.race.values
cat_cols = ['sex', 'race']
num_cols = ['age', 'juv_fel_count', 'juv_misd_count',
            'priors_count', 'is_recid']
cat_sparse = pd.concat([tools.sparsify(compas[c],
                                       long_names=True)
                        for c in cat_cols], axis=1)
X = pd.concat([cat_sparse, 
               compas[num_cols]], 
              axis=1).values
y = gap_cut
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_
preds = np.array([risk_levels[i] for i in np.argmax(probs, axis=1)])
'''

# And balancing the cp scores against the RF
b = balancers.MulticlassBalancer(np.array([p for p in y]), 
                                 np.array([s for s in score_text]), 
                                 race)

# Odds with macro loss
b.adjust(goal='odds', loss='macro')
b.summary()
b.plot()

# Equalized odds with micro loss
b.adjust(goal='odds', loss='micro')
b.summary()
b.plot()

# Strict goal with macro loss
b.adjust(goal='strict', loss='macro')
b.summary()
b.plot()

'''Part 2: Risk stuff with COMPAS'''
# Setting up the data again, only this time with non-recidivists included
compas = compas_full[(compas_full.end - compas_full.start >= 0) &
                ([s in risk_levels for s in compas_full.score_text]) &
                ([s in races for s in compas_full.race])]
compas = compas.reset_index(drop=True)
score = (compas.decile_score / 10).values
score_text = compas.score_text.astype('category')
score_text = score_text.cat.reorder_categories(risk_levels)
race = compas.race.values

# Prepping data for the random forest
cat_cols = ['sex', 'race']
num_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'priors_count']
cat_sparse = pd.concat([tools.sparsify(compas[c],
                                       long_names=True)
                        for c in cat_cols], axis=1)
X = pd.concat([cat_sparse, 
               compas[num_cols]], 
              axis=1).values
y = compas.event.values

# Training a random forest to predict 2-year recidivism
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_[:, 1]

# Getting some basic stats
rf_stats = tools.clf_metrics(y, probs)
cp_stats = tools.clf_metrics(y, score)

# Combining the predictions for plotting
prob_df = pd.DataFrame(zip(probs, score),
                       columns=['rf', 'compas'])
prob_df['race'] = race
prob_df['event'] = y

# Making some plots
# First by race
fig, ax = plt.subplots(3, 2, sharey=True)
for i, r in enumerate(races):
    for j, measure in enumerate(['rf', 'compas']):
        sns.kdeplot(x=measure,
                    data=prob_df[compas.race == r],
                    hue='event',
                    ax=ax[i, j],
                    fill=True)
        ax[i, j].set_ylabel(r)

fig.suptitle('Probability of 2-year recidivism')
plt.show()

# And then overall
fig, ax = plt.subplots(1, 2, sharey=True)
for i, measure in enumerate(['rf', 'compas']):
    sns.kdeplot(x=measure,
                data=prob_df,
                hue='event',
                ax=ax[i],
                multiple='stack')

fig.suptitle('Probability of 2-year recidivism')
plt.show()

# Pickng the optimal thresholds for the random forest's probabilities
p_range = np.arange(0.01, 1, .01)
p_combos = [c for c in combinations(p_range, 2)]
good = np.where(np.diff(p_combos)>= .2)[0]
good_combos = [p_combos[i] for i in good]

with Pool() as p:
    input = [(probs, c, risk_levels, False) for c in good_combos]
    res = p.starmap(tools.otsu, input)
    p.close()
    p.join()

# Finding the cutpoints that yield the lowest intraclass variance
best_cut = [p for p in good_combos[np.argmin(res)]]

# Thresholding the times to crime with the best cutoffs
prob_cut = pd.qcut(x=probs,
                   q=[0] + best_cut + [1],
                   labels=risk_levels).to_list()
prob_cut = np.array(prob_cut)

# Balancing the COMPAS scores against the random forest
b = balancers.MulticlassBalancer(np.array([p for p in prob_cut]), 
                                 np.array([s for s in score_text]), 
                                 race)

# Odds with macro loss
b.adjust(goal='odds', loss='macro')
b.summary()
b.plot()

# Equalized odds with micro loss
b.adjust(goal='odds', loss='micro')
b.summary()
b.plot()

# Strict goal with macro loss
b.adjust(goal='strict', loss='macro')
b.summary()
b.plot()

