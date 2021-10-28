'''Produces the fairness-discrimination plots'''
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import tools
import balancers


# Reading in the original datasets
bar = pd.read_csv('data/source/bar.csv')
weed = pd.read_csv('data/source/drugs.csv')
obesity = pd.read_csv('data/source/obesity.csv')
compas = pd.read_csv('data/source/compas.csv')
park = pd.read_csv('data/source/park.csv')

# Reading in the individual fairness-discrimination grids
bar_fds = pd.read_csv('data/fd_grids/bar.csv')
obesity_fds = pd.read_csv('data/fd_grids/obesity_fds.csv')
compas_fds = pd.read_csv('data/fd_grids/compas_fds.csv')
weed_fds = pd.read_csv('data/fd_grids/cannabis_fds.csv')
park_fds = pd.read_csv('data/fd_grids/park.csv')

dfs = [
    compas_fds, weed_fds, obesity_fds, 
    bar_fds, park_fds
]
df_names = [
    'compas', 'cannabis', 'obesity', 
    'bar', 'parkinsons'
]

# Stacking everything to feed to relpot
for i, df in enumerate(dfs):
    df['dataset'] = [df_names[i]] * df.shape[0]

all_dfs = pd.concat(dfs, axis=0)
losses = ['micro', 'macro']

# Making the two relplots: one for micro loss, and one for macro
sns.set_style('darkgrid')
for i, loss in enumerate(losses):
    df = all_dfs[[loss in s for s in all_dfs.setup]]
    fig, axes = plt.subplots(nrows=1, 
                             ncols=len(df_names))
    
    for i, ax in enumerate(axes):
        ds_df = df[df.dataset == df_names[i]]
        sns.lineplot(x='slack',
                     y=loss + '_loss',
                     hue='setup',
                     data=ds_df,
                     ci=None,
                     ax=ax,
                     palette='colorblind',
                     legend=False)
        ax.set(xlabel='slack',
               ylabel='loss',
               title=df_names[i])
    
    fig.suptitle(loss + ' loss as a function of slack')
    fig.set_tight_layout(True)
    plt.show()
