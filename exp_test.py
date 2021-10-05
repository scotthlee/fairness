import numpy as np
import pandas as pd

from balancers import MulticlassBalancer
from itertools import permutations, combinations
from importlib import reload
from multiprocessing import Pool
from time import time

import tools


# Trying a loop with multiprocessing
outcomes = ['yes', 'no', 'maybe']
groups2 = ['a', 'b']
groups3 = ['a', 'b', 'c']
losses = ['micro', 'macro']
goals = ['odds', 'opportunity', 'strict']

# Setting up the combinations of situations
pred_types = ['low', 'medium', 'high']
out_types = ['balance', 'equal_imbalance', 'unequal_imbalance']
group_types = ['no_minority', 'slight_minority', 'strong_minority']

sits = [[[[g, o, p] for p in pred_types]
             for o in out_types]
            for g in group_types]
sits = tools.flatten([l for l in tools.flatten(sits)])

# Trying a run for a 3-class 2-group problem
p23 = {'groups': {
           'slight_minority': np.array([.3, .7]),
           'no_minority': np.array([.5, .5]),
           'strong_minority': np.array([.2, .8])},
       'outcomes': {
           'balance': np.array([[.333, .333, .334],
                                [.333, .333, .334]]),
           'equal_imbalance': np.array([[.1, .5, .4],
                                        [.1, .5, .4]]),
           'unequal_imbalance': np.array([[.1, .5, .4],
                                          [.4, .5, .1]])},
       'preds': {
           'low': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                          np.array([[.7, .15, .15],
                                    [.15, .7, .15],
                                    [.15, .15, .7]])],
           'medium': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                        np.array([[.5, .25, .25],
                                  [.25, .5, .25],
                                  [.25, .25, .5]])],
           'high': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                         np.array([[.3, .3, .4],
                                   [.4, .3, .3],
                                   [.4, .3, .3]])]
           }
       }

input_23 = [[[(p23['groups'][g], p23['outcomes'][o], p23['preds'][p]) 
        for p in pred_types] 
       for o in out_types] 
      for g in group_types]
input_23 = tools.flatten([l for l in tools.flatten(input_23)])

# Running the sim
with Pool() as p:
    input = [[[(outcomes, 
                groups2, 
                t[0], 
                t[1], 
                t[2], 
                loss, 
                goal,
                sits[i][0],
                sits[i][1],
                sits[i][2])
            for i, t in enumerate(input_23)]
           for loss in losses]
          for goal in goals]
    input = tools.flatten([l for l in tools.flatten(input)])
    res = p.starmap(tools.test_run, input)
    p.close()
    p.join()

stats_23 = pd.concat([r['stats'] for r in res], axis=0)
stats_23.to_csv('2-group 3-class stats.csv', index=False)

# Setting up a 3-group 3-group problem
p33 = {'outcomes': {
            'balance': [[.333, .333, .334],
                        [.333, .333, .334],
                [.333, .333, .334]],
    'equal_imbalance': [[.1, .5, .4],
                        [.1, .5, .4],
                        [.1, .5, .4]],
    'unequal_imbalance': [[.1, .5, .4],
                          [.5, .4, .1],
                          [.5, .4, .1]]
    },
          'groups': {
               'no_minority': [.333, .333, .334],
               'one_slight_minority': [.2, .4, .4],
               'two_slight_minorities': [.2, .6, .2],
               'two_strong_minorities': [.1, .8, .1],
               'one_strong_minority': [.1, .5, .4]
          }
    }

