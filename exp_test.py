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
p32 = tools.prob_perms(3, 2)
p32_11_input = [(outcomes, groups, l[0], l[1], l[2], 'macro') 
                for l in p32[1][1]]

with Pool() as p:
    s = time()
    res = p.starmap(tools.test_run, p32_11_input)
    f = time()
    p.close()
    p.join()

e = f - s

