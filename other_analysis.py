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


# Options for balancing
setups = [
    ['odds', 'macro'], ['odds', 'micro'],
    ['strict', 'macro'], ['strict', 'micro'],
    ['opportunity', 'macro'], ['opportunity', 'micro'],
    ['demographic_parity', 'macro'], ['demographic_parity', 'micro']
]

'''Part 1: Drug use data'''
# Loading the data
drugs = pd.read_csv('data/source/drugs.csv')

# Setting race as white vs. non-white
race = np.array(['white' if e == -0.31685 else 'non-white' 
                 for e in drugs.ethnicity])
drugs['race']= race
races = ['non-white', 'white']

# Setting the target to meth with some collapsed categories
weed = drugs.cannabis.astype(str)
y = np.array(['never'] * len(weed), dtype='<U20')
not_last_year = ['CL1', 'CL2']
last_year = ['CL3', 'CL4', 'CL5', 'CL6']
for i, o in enumerate(y):
    if weed[i] in ['CL1', 'CL2']:
        y[i] = 'not last year'
    elif weed[i] in ['CL3', 'CL4', 'CL5', 'CL6']:
        y[i] = 'last year'

outcomes = np.unique(y)

# Making a sparse matrix of predictors for the random forest
input_cols = [
    'age', 'gender', 'education',
    'country', 'nscore', 'escore',
    'oscore', 'cscore', 'impulsive',
    'alcohol', 'nicotine', 'caff',
    'choc', 'mushrooms'
]
X = pd.concat([tools.sparsify(drugs[c].astype(str))
               for c in input_cols], axis=1).values

# Running the random forest
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_
preds = np.array([np.unique(y)[i] for i in np.argmax(probs, axis=1)])
bin_preds = [np.array([preds == o], dtype=np.uint8) for o in outcomes]

# Getting some basic stats
rf_stats = tools.clf_metrics(y, preds)

# Making a few basic density plots
sns.set_style('darkgrid')
sns.set_palette('colorblind')

# First for everyone
fig, ax = plt.subplots(3, sharey=True)
for i, o in enumerate(outcomes):
    sns.kdeplot(x=probs[:, i],
                hue=np.array(preds == o, dtype=np.uint8),
                ax=ax[i],
                fill=True)
    ax[i].set_ylabel(o)

fig.suptitle('Predicted probability of cannabis usage')
plt.show()

# And then by race
fig, ax = plt.subplots(2, 3, sharey=True, sharex=True)
for i, r in enumerate(races):
    ids = np.where(race == r)[0]
    for j, o in enumerate(outcomes):
        sns.kdeplot(x=probs[ids, j],
                    hue=bin_preds[j][0][ids],
                    ax=ax[i, j],
                    fill=True)
        ax[i, j].set_ylabel(r)
        ax[i, j].set_xlabel(o)

fig.suptitle('Predicted probability of cannabis use by race')
plt.show()

tables = []
fds = []

b = balancers.MulticlassBalancer(y, preds, race)
for i, setup in enumerate(setups):
    goal, loss = setup[0], setup[1]
    title = goal + ' goal with ' + loss + ' loss'
    b.adjust_new(goal=goal, loss=loss)
    b.plot(title=title, 
           tight=True, 
           show=False,
           save=True,
           img_dir='img/weed/')
    tables.append(tools.cp_mat_summary(b,
                                       title=title,
                                       slim=(i>0)))
    fds.append(tools.fd_grid(b,
                             loss=loss,
                             goal=goal,
                             step=.001,
                             max=.15))

pd.concat(tables, axis=1).to_csv('tables/weed.csv', index=False)

# Making the fairness-discrimination plot
for i, df in enumerate(fds):
    setup = setups[i][0] + ' ' + setups[i][1]
    df['setup'] = [setup] * df.shape[0]

# Making a df for plotting
all_fds = pd.concat(fds, axis=0)
all_fds.to_csv('data/fd_grids/cannabis.csv', index=False)

'''Part 2: Obesity data'''
# Iporting the data
obesity_full = pd.read_csv('data/source/obesity.csv')

# Filtering out classes with high gender imbalance
good_rows = np.where([w not in ['Obesity_Type_II',
                                'Obesity_Type_III']
                      for w in obesity_full.NObeyesdad.values])[0]
obesity = obesity_full.iloc[good_rows, :].reset_index(drop=True)

# Setting up the outcome and protected attribute
gender = obesity.Gender.values
genders = np.unique(gender)
weight_class = obesity.NObeyesdad.values
weight_classes = np.unique(weight_class)
n_classes = len(weight_classes)
n_groups = len(genders)

# Making a sparse matrix of predictors for the random forest
cat_cols = [
    'Gender', 'family_history_with_overweight', 'FAVC',
    'CAEC', 'SMOKE', 'SCC', 
    'CALC', 'MTRANS'
]
num_cols = [
    'Age', 'FCVC', 'NCP',
    'CH2O', 'FAF', 'TUE',
]
X_cat = pd.concat([tools.sparsify(obesity[c].astype(str),
                                  long_names=True)
               for c in cat_cols], axis=1)
X = pd.concat([X_cat, obesity[num_cols].round()], axis=1)
y = weight_class

# Running the random forest
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True)
rf.fit(X, y)
probs = rf.oob_decision_function_
preds = np.array([np.unique(y)[i] for i in np.argmax(probs, axis=1)])
bin_preds = [np.array([preds == w], dtype=np.uint8) 
             for w in weight_classes]

# Getting some basic stats
rf_stats = tools.clf_metrics(y, preds)

# Making a few basic density plots
sns.set_style('darkgrid')
sns.set_palette('colorblind')

# First for everyone
fig, ax = plt.subplots(nrows=1, 
                       ncols=n_classes,
                       sharey=True)
for i, o in enumerate(weight_classes):
    sns.kdeplot(x=probs[:, i],
                hue=np.array(preds == o, dtype=np.uint8),
                ax=ax[i],
                fill=True)
    ax[i].set_xlabel(o)
    ax[i].set_ylabel('')

fig.suptitle('Predicted probability of weight category')
fig.set_tight_layout(True)
plt.show()

# And then by race
fig, ax = plt.subplots(n_groups, 
                       n_classes, 
                       sharey=True, 
                       sharex=True)
for i, g in enumerate(genders):
    ids = np.where(gender == g)[0]
    for j, o in enumerate(weight_classes):
        sns.kdeplot(x=probs[ids, j],
                    hue=bin_preds[j][0][ids],
                    ax=ax[i, j],
                    fill=True)
        ax[i, j].set_ylabel(r)
        ax[i, j].set_xlabel(o)

fig.suptitle('Predicted probability of weight category use by gender')
plt.show()

tables = []
fds = []

b = balancers.MulticlassBalancer(y, preds, gender)
for i, setup in enumerate(setups):
    goal, loss = setup[0], setup[1]
    title = goal + ' goal with ' + loss + ' loss'
    b.adjust_new(goal=goal, loss=loss)
    b.plot(title=title, 
           tight=True, 
           show=False,
           save=True,
           img_dir='img/obesity/')
    tables.append(tools.cp_mat_summary(b,
                                       title=title,
                                       slim=(i>0)))
    
    fds.append(tools.fd_grid(b,
                             loss=loss,
                             goal=goal,
                             step=.001,
                             max=.15))

pd.concat(tables, axis=1).to_csv('data/tables/obesity tables.csv', 
                                 index=False)

# Making the fairness-discrimination plot
for i, df in enumerate(fds):
    setup = setups[i][0] + ' ' + setups[i][1]
    df['setup'] = [setup] * df.shape[0]

# Making a df for plotting
all_fds = pd.concat(fds, axis=0)
all_fds.to_csv('data/fd_grids/obesity.csv', index=False)

'''Part 3: Bar passage data'''
# Importing the data and setting the classification target
bar = pd.read_csv('data/source/bar.csv')
bar = bar[bar.bar != 'e non-Grad']
bar = bar.reset_index(drop=True)
passed = bar.bar.values

# Making a set of predictors for the random forest
# Setting up the outcome and protected attribute
race = bar.race.values
white = np.array(['white' if r == 7.0 else 'non-white' for r in race])
races = np.unique(race)
pass_classes = np.unique(passed)

n_classes = len(pass_classes)
n_groups = len(races)

# Making a sparse matrix of predictors for the random forest
cat_cols = [
    'sex', 'race'
]
num_cols = [
    'lsat', 'ugpa', 'gpa'
]
X_cat = pd.concat([tools.sparsify(bar[c].astype(str),
                                  long_names=True)
               for c in cat_cols], axis=1)
X = pd.concat([X_cat, bar[num_cols].round()], axis=1)
y = passed

# Running the random forest
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True,
                            class_weight='balanced')
rf.fit(X, y)
probs = rf.oob_decision_function_
prob_args = np.argmax(probs, axis=1)
preds = np.array([pass_classes[i] for i in prob_args])
bin_preds = [np.array([preds == p], dtype=np.uint8) 
             for p in pass_classes]

# Getting some basic stats
rf_stats = tools.clf_metrics(y, preds)

tables = []
fds = []

# Running the balancing loop
b = balancers.MulticlassBalancer(y, preds, white)
for i, setup in enumerate(setups):
    goal, loss = setup[0], setup[1]
    title = goal + ' goal with ' + loss + ' loss'
    '''
    b.adjust_new(goal=goal, loss=loss)
    b.plot(title=title, 
           tight=True, 
           show=False,
           save=True,
           img_dir='img/bar/')
    tables.append(tools.cp_mat_summary(b,
                                       title=title,
                                       slim=(i>0)))'''
    
    fds.append(tools.fd_grid(b,
                             loss=loss,
                             goal=goal,
                             step=.001,
                             max=.15))

# Making the fairness-discrimination plot
for i, df in enumerate(fds):
    setup = setups[i][0] + ' ' + setups[i][1]
    df['setup'] = [setup] * df.shape[0]

# Making a df for plotting
all_fds = pd.concat(fds, axis=0)
all_fds.to_csv('data/fd_grids/bar.csv', index=False)

# Doing the plots
fig, ax = plt.subplots(nrows=1, 
                       ncols=n_classes,
                       sharey=True)
for i, p in enumerate(pass_classes):
    sns.kdeplot(x=probs[:, i],
                hue=np.array(preds == p, dtype=np.uint8),
                ax=ax[i],
                fill=True)
    ax[i].set_xlabel(p)
    ax[i].set_ylabel('')

fig.suptitle('Predicted probability of bar passage')
fig.set_tight_layout(True)
plt.show()

# And then by race
fig, ax = plt.subplots(2, 
                       n_classes, 
                       sharey=True, 
                       sharex=True)
for i, r in enumerate(np.unique(white)):
    ids = np.where(white == r)[0]
    for j, p in enumerate(pass_classes):
        sns.kdeplot(x=probs[ids, j],
                    hue=bin_preds[j][0][ids],
                    ax=ax[i, j],
                    fill=True)
        ax[i, j].set_ylabel(r)
        ax[i, j].set_xlabel(p)

fig.suptitle('Predicted probability of bar passage by race')
plt.show()
