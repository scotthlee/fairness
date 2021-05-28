'''Generates the synthetic dataset used for the demo notebook'''
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


# Simulating a multiclass outcome
cat_m_roc = np.array([[.7, .1, .2],
                      [.2, .6, .2],
                      [.2, .3, .5]])
dog_m_roc = np.array([[.4, .3, .3],
                      [.1, .8, .1],
                      [.1, .2, .7]])
sheep_m_roc = np.array([[.3, .6, .1],
                        [.15, .7, .15],
                        [.1, .1, .8]])
m_rocs = np.array([cat_m_roc, 
                   dog_m_roc, 
                   sheep_m_roc])

cat_probs = [.5, .4, .1]
dog_probs = [.6, .3, .1]
sheep_probs = [.4, .2, .4]
probs = [cat_probs, dog_probs, sheep_probs]

actions = ['feed', 'pet', 'shear']
animals = ['cat', 'dog', 'sheep']

a = tools.make_catvar(n=1000, 
                      p=[.4, .4, .2], 
                      levels=animals)
a_ids = [np.where(a == c)[0] for c in animals]
y_gen = tools.make_label(p=probs,
                         catvar=a,
                         levels=actions)

# Combining the two into a data frame and sorting
df = pd.DataFrame((a, y_gen),
                  index=['animal', 'action']).T

# Resetting the target array after sorting
y = df.action.values
y_ = tools.make_multi_predictor(y, m_rocs, a)
df['pred_action'] = y_

# Making the shearing probabilities to do ROC stuff
n = norm(0, 1)
cat_gens = [norm(-.8, 1.3), norm(.8, 1)]
dog_gens = [norm(-.5, 1.2), norm(.5, .8)]
sheep_gens = [norm(-1.3, 1), norm(1.3, 1)]
gen_dict = {'cat': cat_gens,
            'dog': dog_gens,
            'sheep': sheep_gens}

for a in animals:
    n1 = df[df.animal == a].shear.sum()
    n0 = np.sum(df.animal == a) - n1
    s0 = gen_dict[a][0].rvs(n0)
    s1 = gen_dict[a][1].rvs(n1)
    df.shear_prob[(df.animal == a) & (df.shear == 0)] = n.cdf(s0)
    df.shear_prob[(df.animal == a) & (df.shear == 1)] = n.cdf(s1)


# Saving to disk
df.to_csv('pets.csv', index=False)
