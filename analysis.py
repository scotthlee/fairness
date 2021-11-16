import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from matplotlib import pyplot as plt

import tools
import balancers


data_dir = 'data/source/'

# Reading in the original datasets
bar = pd.read_csv(data_dir + 'bar.csv')
weed = pd.read_csv(data_dir + 'drugs.csv')
obesity = pd.read_csv(data_dir + 'obesity.csv')
park = pd.read_csv(data_dir + 'park.csv')

df_names = ['bar', 'cannabis', 'obesity', 'parkinsons']
datasets = [bar, weed, obesity, park]

# Getting the post-adjustment results for each dataset
pop_stats = []
cv_stats = []

for i, ds in enumerate(datasets):
    b = balancers.MulticlassBalancer(ds.y.values,
                                     ds.yhat.values,
                                     ds.a.values)
    b.adjust(goal='strict',
             loss='macro')
    pop_stats.append(tools.balancing_stats(b))
    b.adjust(goal='strict',
             loss='macro',
             cv=True,
             shuffle=True,
             seed=2021)
    cv_stats.append(tools.balancing_stats(b))

pop_df = pd.concat(pop_stats, axis=0)
pop_df['dataset'] = df_names
pop_df['type'] = 'pop'

cv_df = pd.concat(cv_stats, axis=0)
cv_df['dataset'] = df_names
cv_df['type'] = 'cv'

out = pd.concat([pop_df, cv_df], axis=0)
out.to_csv('data/tables/balancing_stats.csv',
           index=False)

