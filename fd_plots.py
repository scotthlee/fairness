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
park = pd.read_csv('data/source/park.csv')

df_names = ['bar', 'cannabis', 'obesity', 'parkinsons']

datasets = [bar, weed, obesity, park]

# Defining the 4 fairness types
goals = ['odds', 'strict', 'opportunity', 'demographic_parity']

# Making the two relplots: one for micro loss, and one for macro
sns.set_style('darkgrid')
sns.set(font_scale=.75)

fig, axes = plt.subplots(nrows=4, ncols=4, sharey=False)

for i, goal in enumerate(goals):
    for j, df in enumerate(datasets):
        b = balancers.MulticlassBalancer(df.y.values,
                                         df.yhat.values,
                                         df.a.values)
        grid = tools.fd_grid(b, 
                             step=.01,
                             loss='macro', 
                             goal=goal)
        tools.fd_plot(grid,
                      goal=goal,
                      ax=axes[i, j])

# Labeling the fairness constraint types
for i, goal in enumerate(goals):
    plt.setp(axes[i, 0], ylabel=goal)

# Labeling the datasets
for i, ds in enumerate(df_names):
    plt.setp(axes[3, i], xlabel=ds)

fig.set_tight_layout(True)
plt.suptitle('Fairness vs. discrimination')
plt.show()
