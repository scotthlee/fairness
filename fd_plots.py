import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import tools
import balancers


# Reading in the original datasets
weed = pd.read_csv('data/drugs.csv')
obesity = pd.read_csv('data/obesity.csv')
compas = pd.read_csv('data/compas.csv')

# Reading in the individual fairness-discrimination grids
obesity_fds = pd.read_csv('data/obesity_fds.csv')
compas_fds = pd.read_csv('data/compas_fds.csv')
weed_fds = pd.read_csv('data/cannabis_fds.csv')
dfs = [compas_fds, weed_fds, obesity_fds]
df_names = ['compas', 'cannabis', 'obesity']

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
    #fig.set_tight_layout(True)
    plt.show()
