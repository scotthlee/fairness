'''Generates some visuals from the forest-based probabilities'''

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


data_dir = 'data/source/'

# Reading in the original datasets
bar = pd.read_csv(data_dir + 'bar.csv')
weed = pd.read_csv(data_dir + 'drugs.csv')
obesity = pd.read_csv(data_dir + 'obesity.csv')
park = pd.read_csv(data_dir + 'park.csv')

# And reading in the predicted probabilities
bar_probs = pickle.load(open(data_dir + 'bar_probs.pkl', 'rb'))
weed_probs = pickle.load(open(data_dir + 'weed_probs.pkl', 'rb'))
obesity_probs = pickle.load(open(data_dir + 'obesity_probs.pkl', 'rb'))
park_probs = pickle.load(open(data_dir + 'park_probs.pkl', 'rb'))

df_names = ['bar', 'cannabis', 'obesity', 'parkinsons']
datasets = [bar, weed, obesity, park]
probs = [bar_probs, weed_probs, obesity_probs, park_probs]

titles = [
    'bar passage', 'cannabis usage',
    'obesity', 'parkinsons'
]


for i, ds in enumerate(datasets):
    groups = ds.a.values
    group_levels = np.unique(groups)
    outcomes = np.unique(ds.y.values)
    n_groups = len(group_levels)
    n_outcomes = len(np.unique(ds.y.values))
    
    # Gtting the binary-level predictions
    preds = ds.y.values
    bin_preds = [np.array([ds.y.values == o for o in outcomes],
                          dtype=np.uint8)]
    
    # Making a few basic density plots
    sns.set_style('darkgrid')
    sns.set_palette('colorblind')
    
    # First for everyone
    fig, ax = plt.subplots(n_outcomes, sharey=True)
    for j, o in enumerate(outcomes):
        sns.kdeplot(x=probs[i][:, j],
                    hue=np.array(preds == o, dtype=np.uint8),
                    ax=ax[j],
                    fill=True)
        ax[j].set_ylabel(o)
    
    fig.suptitle('Predicted probability of ' + titles[i])
    plt.show()
    
    # And then by group
    fig, ax = plt.subplots(n_groups, 
                           n_outcomes, 
                           sharey=True, 
                           sharex=True)
    for j, g in enumerate(group_levels):
        ids = np.where(groups == g)[0]
        for k, o in enumerate(outcomes):
            sns.kdeplot(x=probs[i][ids, k],
                        hue=bin_preds[k][0][ids],
                        ax=ax[j, k],
                        fill=True)
            ax[j, g].set_ylabel(g)
            ax[j, g].set_xlabel(o)
    
    fig.suptitle('Predicted probability of ' + titles[i])
    plt.show()
