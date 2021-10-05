import numpy as np
import pandas as pd

from balancers import MulticlassBalancer
from itertools import permutations, combinations
from multiprocessing import Pool
from time import time

import tools

'''
outcomes = [1, 2, 3]
groups = ['a', 'b', 'c']
p_group = [.3, .7]
p_y_group = [[.1, .1, .8],
             [.3, .5, .2]]
p_yh_group = np.array([[[.1, .2, .7], 
                   [.6, .3, .1], 
                   [.1, .1, .8]],
                  [[.8, .1, .1], 
                   [.2, .5, .3], 
                   [.2, .2, .6]]])
test = test_run(outcomes,
                groups,
                p_group,
                p_y_group,
                p_yh_group)
'''
# Trying a loop with multiprocessing
outcomes = ['yes', 'no', 'maybe']
groups = ['a', 'b']

# Options for p_group
p23 = {'groups': {
           'slight_minority': np.array([.3, .7]),
           'no_minority': np.array([.5, .5]),
           'strong_minority': np.array([.2, .8])},
       'outcomes': {
           'balance': np.array([[.333, .333, .334],
                                [.333, .333, .334]]),
           'equal_imblanace': np.array([[.1, .5, .4],
                                        [.1, .5, .4]]),
           'unequal_imblanace': np.array([[.1, .5, .4],
                                          [.4, .5, .1]])},
       'preds': {
           'great_good': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                          np.array([[.7, .15, .15],
                                    [.15, .7, .15],
                                    [.15, .15, .7]])],
           'great_ok': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                        np.array([[.5, .25, .25],
                                  [.25, .5, .25],
                                  [.25, .25, .5]])],
           'great_bad': [np.array([[.9, .05, .05],
                                   [.05, .9, .05],
                                   [.05, .05, .9]]),
                         np.array([[.3, .3, .4],
                                   [.4, .3, .3],
                                   [.4, .3, .3]])]
           }
       }

# Setting up the combinations of 2 groups and 3 outcomes
c23 = [[[(p23['groups'][g], p23['outcomes'][o], p23['preds'][p]) 
        for p in p23['preds'].keys()] 
       for o in p23['outcomes'].keys()] 
      for g in p23['groups'].keys()]
c23 = tools.flatten([l for l in tools.flatten(c)])

# Running the sim
res23 = [tools.test_run(outcomes, groups, p[0], p[1], p[2]) for p in c]

p32 = {'groups': {
           'no_minority': [.333, .333, .334],
           'one_slight_minority': [.2, .4, .4],
           'two_slight_minorities': [.2, .6, .2],
           'two_strong_minorities': [.1, .8, .1],
           'one_strong_minority': [.1, .5, .4]
            },
       'outcomes': {
           'balance': [[.333, .333, .334],
                       [.333, .333, .334]],
           'equal_imblanace': [[.1, .5, .4],
                               [.1, .5, .4]],
           'unequal_imblanace': [[.1, .5, .4],
                                 [.4, .5, .1]]}
        }

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

