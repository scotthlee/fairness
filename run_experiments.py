import numpy as np
import pandas as pd
import argparse
import pickle
import os

from itertools import permutations, combinations
from importlib import reload
from multiprocessing import Pool
from time import time

import balancers
import tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',
                        type=int,
                        default=1,
                        help='number of times to run the experiments')
    args = parser.parse_args()
    
    # Setting the globals
    seeds = np.random.randint(0, 1e6, args.runs)
    
    # Trying a loop with multiprocessing
    outcomes = ['yes', 'no', 'maybe']
    groups2 = ['a', 'b']
    groups3 = ['a', 'b', 'c']
    losses = ['micro', 'macro']
    goals = ['odds', 'opportunity', 'strict', 'demographic_parity']

    # Setting up the combinations of situations
    bias_types = ['low', 'medium', 'high']
    out_types = ['balanced', 'one_rare', 'two_rare']
    group_types = ['no_minority', 'slight_minority', 'strong_minority']

    sits = [[[[g, o, p] for p in bias_types]
                 for o in out_types]
                for g in group_types]
    sits = tools.flatten([l for l in tools.flatten(sits)])

    # Trying a run for a 3-class 2-group problem
    p23 = {'groups': {
            'no_minority': np.array([.5, .5]),
            'slight_minority': np.array([.7, .3]),
            'strong_minority': np.array([.9, .1])},
           'outcomes': {
               'balanced': np.array([[.333, .333, .334],
                                    [.333, .333, .334]]),
               'one_rare': np.array([[.1, .5, .4],
                                     [.1, .5, .4]]),
               'two_rare': np.array([[.1, .8, .1],
                                     [.1, .8, .1]]),
               },
           'bias': {
               'low': [np.array([[.9, .05, .05],
                                 [.05, .9, .05],
                                 [.05, .05, .9]]),
                       np.array([[.8, .1, .1],
                                 [.1, .8, .1],
                                 [.1, .1, .8]])],
               'medium': [np.array([[.9, .05, .05],
                                       [.05, .9, .05],
                                       [.05, .05, .9]]),
                            np.array([[.7, .15, .15],
                                      [.15, .7, .15],
                                      [.15, .15, .7]])],
               'high': [np.array([[.9, .05, .05],
                                       [.05, .9, .05],
                                       [.05, .05, .9]]),
                             np.array([[.5, .25, .25],
                                       [.25, .5, .25],
                                       [.25, .25, .5]])]
               }
           }

    input_23 = [[[(p23['groups'][g], p23['outcomes'][o], p23['bias'][b]) 
            for b in bias_types] 
           for o in out_types] 
          for g in group_types]
    input_23 = tools.flatten([l for l in tools.flatten(input_23)])

    # Running the sim
    stats_23 = []
    for seed in seeds:
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
                        sits[i][2],
                        seed)
                    for i, t in enumerate(input_23)]
                   for loss in losses]
                  for goal in goals]
            input = tools.flatten([l for l in tools.flatten(input)])
            res = p.starmap(tools.test_run, input)
            p.close()
            p.join()
    
            #rocs_23 = [[r['old_rocs'], r['new_rocs']] for r in res]
            df = pd.concat([r['stats'] for r in res], axis=0)
            df['n_groups'] = 2
            
            if 'exp_stats.csv' in os.listdir():
                mode = 'a'
            else: mode = 'w'
            
            df.to_csv('data/exp_stats.csv', mode=mode, index=False)
    
    # Setting up a 3-group 3-group problem
    bias_types = ['low_one', 'medium_one', 'high_one',
                  'low_two', 'medium_two', 'high_two']
    out_types = ['balanced', 'one_rare', 'two_rare']
    group_types = ['no_minority', 'one_slight_minority', 'one_strong_minority',
                   'two_slight_minorities', 'two_strong_minorities']

    sits = [[[[g, o, p] for p in bias_types]
                 for o in out_types]
                for g in group_types]
    sits = tools.flatten([l for l in tools.flatten(sits)])

    p33 = {'groups': {
            'no_minority': np.array([.33, .33, .34]),
            'one_slight_minority': np.array([.4, .4, .2]),
            'one_strong_minority': np.array([.45, .45, .1]),
            'two_slight_minorities': np.array([.6, .2, .2]),
            'two_strong_minorities': np.array([.8, .1, .1])},
           'outcomes': {
               'balanced': np.array([[.333, .333, .334],
                                    [.333, .333, .334],
                                    [.333, .333, .334]]),
               'one_rare': np.array([[.1, .5, .4],
                                     [.1, .5, .4],
                                     [.1, .5, .4]]),
               'two_rare': np.array([[.1, .8, .1],
                                     [.1, .8, .1],
                                     [.1, .8, .1]]),
               },
           'bias': {
               'low_one': [np.array([[.9, .05, .05],
                                     [.05, .9, .05],
                                     [.05, .05, .9]]),
                           np.array([[.9, .05, .05],
                                     [.05, .9, .05],
                                     [.05, .05, .9]]),
                            np.array([[.8, .1, .1],
                                      [.1, .8, .1],
                                      [.1, .1, .8]])],
               'medium_one': [np.array([[.9, .05, .05],
                                       [.05, .9, .05],
                                       [.05, .05, .9]]),
                              np.array([[.9, .05, .05],
                                        [.05, .9, .05],
                                        [.05, .05, .9]]),
                            np.array([[.7, .15, .15],
                                      [.15, .7, .15],
                                      [.15, .15, .7]])],
               'high_one': [np.array([[.9, .05, .05],
                                       [.05, .9, .05],
                                       [.05, .05, .9]]),
                            np.array([[.9, .05, .05],
                                      [.05, .9, .05],
                                      [.05, .05, .9]]),
                             np.array([[.5, .25, .25],
                                       [.25, .5, .25],
                                       [.25, .25, .5]])],
                 'low_two': [np.array([[.9, .05, .05],
                                       [.05, .9, .05],
                                       [.05, .05, .9]]),
                             np.array([[.8, .1, .1],
                                       [.1, .8, .1],
                                       [.1, .1, .8]]),
                              np.array([[.8, .1, .1],
                                        [.1, .8, .1],
                                        [.1, .1, .8]])],
                 'medium_two': [np.array([[.9, .05, .05],
                                         [.05, .9, .05],
                                         [.05, .05, .9]]),
                                np.array([[.7, .15, .15],
                                          [.15, .7, .15],
                                          [.15, .15, .7]]),
                              np.array([[.7, .15, .15],
                                        [.15, .7, .15],
                                        [.15, .15, .7]])],
                 'high_two': [np.array([[.9, .05, .05],
                                         [.05, .9, .05],
                                         [.05, .05, .9]]),
                              np.array([[.5, .25, .25],
                                        [.25, .5, .25],
                                        [.25, .25, .5]]),
                               np.array([[.5, .25, .25],
                                         [.25, .5, .25],
                                         [.25, .25, .5]])]
               }
           }

    input_33 = [[[(p33['groups'][g], p33['outcomes'][o], p33['bias'][b]) 
            for b in bias_types] 
           for o in out_types] 
          for g in group_types]
    input_33 = tools.flatten([l for l in tools.flatten(input_33)])

    # Running the sim
    stats_33 = []
    for seed in seeds:
        with Pool() as p:
            input = [[[(outcomes, 
                        groups3, 
                        t[0], 
                        t[1], 
                        t[2], 
                        loss, 
                        goal,
                        sits[i][0],
                        sits[i][1],
                        sits[i][2],
                        seed)
                    for i, t in enumerate(input_33)]
                   for loss in losses]
                  for goal in goals]
            input = tools.flatten([l for l in tools.flatten(input)])
            res = p.starmap(tools.test_run, input)
            p.close()
            p.join()

            #rocs_33 = [[r['old_rocs'], r['new_rocs']] for r in res]
            df = pd.concat([r['stats'] for r in res], axis=0)
            df['n_groups'] = 3
            df.to_csv('data/exp_stats.csv', 
                      mode='a', 
                      header=False,
                      index=False)
    
    #stats_33 = pd.concat(stats_33, axis=0)
    
    # Putting the 2 together and saving to disk
    #stats = pd.concat([stats_23, stats_33], axis=0)
    #stats.to_csv('data/exp_stats.csv', mode='a', index=False)
