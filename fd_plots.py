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

# Also separating by objective function in case that's needed
micro = all_dfs[['micro' in s for s in all_dfs.setup]]
micro.setup = [s.replace(' micro', '') for s in micro.setup.values]
macro = all_dfs[['macro' in s for s in all_dfs.setup]]
macro.setup = [s.replace(' macro', '') for s in macro.setup.values]
obj_dfs = [micro, macro]

# Making the two relplots: one for micro loss, and one for macro
sns.set_style('darkgrid')
sns.set(font_scale=.75)

for i, loss in enumerate(losses):
    df = obj_dfs[i]
    fig, axes = plt.subplots(nrows=2, ncols=3)
    
    for i, ax in enumerate(tools.flatten(axes)):
        if i == len(df_names):
            ax.axis('off')
        else:
            ds_df = df[df.dataset == df_names[i]]
            sns.lineplot(x='slack',
                         y=loss + '_loss',
                         hue='setup',
                         data=ds_df,
                         ci=None,
                         ax=ax,
                         palette='colorblind',
                         legend='brief')
            ax.set(xlabel='slack',
                   ylabel='loss',
                   title=df_names[i])
    fig.suptitle('Macro loss as a function of slack')
    fig.set_tight_layout(True)
    plt.show()
