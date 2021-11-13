'''Makes random forest-based predictors for the real-world datasets'''
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
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

# List to hold the stats from each balancing run
stats = []

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

# Exporting the predictions
drugs['a'] = race
drugs['y'] = y
drugs['yhat'] = preds
drugs.to_csv('data/source/drugs.csv', index=False)

with open('data/source/weed_probs.pkl', 'wb') as f:
    pickle.dump(probs, f)


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

# Exporting the results
obesity['a'] = gender
obesity['y'] = y
obesity['yhat'] = preds
obesity.to_csv('data/source/obesity.csv', index=False)

with open('data/source/obesity_probs.pkl', 'wb') as f:
    pickle.dump(probs, f)


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

# Exporting the predictions
bar['a'] = white
bar['y'] = y
bar['yhat'] = preds
bar.to_csv('data/source/bar.csv', index=False)

with open('data/source/bar_probs.pkl', 'wb') as f:
    pickle.dump(probs, f)


'''Parkinson's data'''
park = pd.read_csv('data/source/park.csv')
sex = np.array(['female' if s == 1 else 'male' 
                for s in park.sex.values])
score = park.total_UPDRS.values
risk_levels = ['Mild', 'Moderate', 'Severe']

groups = np.unique(sex)
n_groups = len(groups)
n_classes = len(risk_levels)

# Using Otsu thresholding to pick risk levels based on UPDRS score
if 'score_cut' not in park.columns.values:
    p_range = np.arange(0.01, 1, .01)
    p_combos = [c for c in combinations(p_range, 2)]
    good = np.where(np.diff(p_combos)>= .2)[0]
    good_combos = [p_combos[i] for i in good]
    
    with Pool() as p:
        input = [(score, c, risk_levels) for c in good_combos]
        res = p.starmap(tools.otsu, input)
        p.close()
        p.join()
        
    # Selecting the cutoff that yielded the lowest intraclass variance
    best_cut = [p for p in good_combos[np.argmin(res)]]
    
    # Thresholding the times to crime with the best cutoffs
    score_cut = pd.qcut(x=score,
                      q=[0] + best_cut + [1],
                      labels=risk_levels).to_list()
    score_cut = np.array(score_cut)
    park['score_cut'] = score_cut
else:
    score_cut = park.score_cut.values

# Making a tall version for visualization
score_df = pd.DataFrame(zip(score_cut, score), 
                        columns=['dik', 'days'])
score_df['sex'] = sex

n_classes = len(pass_classes)
n_groups = len(races)

# Making a sparse matrix of predictors for the random forest
X = park.drop(['subject#',
               'motor_UPDRS', 
               'total_UPDRS',
               'score_cut'], axis=1)
y = park.score_cut.values

# Running the random forest
rf = RandomForestClassifier(n_jobs=-1, 
                            n_estimators=1000, 
                            oob_score=True,
                            class_weight='balanced')
rf.fit(X, y)
probs = rf.oob_decision_function_
prob_args = np.argmax(probs, axis=1)
preds = np.array([risk_levels[i] for i in prob_args])
bin_preds = [np.array([preds == p], dtype=np.uint8) 
             for p in risk_levels]

with open('data/source/park_probs.pkl', 'wb') as f:
    pickle.dump(probs, f)

# Exporting the predictions
park['a'] = sex
park['y'] = y
park['yhat'] = preds
park.to_csv('data/source/park.csv', index=False)

