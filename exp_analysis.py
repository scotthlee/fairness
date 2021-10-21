import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from multiprocessing import Pool
from matplotlib import pyplot as plt

import tools
import balancers

'''Part 3: Working with the synthetic experiments data'''
# Reading in the  data
exp = pd.read_csv('~/Desktop/exp_stats.csv')

# Separating by n_groups
exp2 = exp[exp.n_groups == 2]
exp3 = exp[exp.n_groups == 3]

# Getting some basic odds ratios for triviality
factors = ['loss', 'goal', 'class_balance', 'group_balance']
triv_ors2 = [[tools.odds_ratio(exp2.trivial, exp2[fac] == f)
              for f in exp2[fac].unique()]
             for fac in factors]
triv_ors3 = [[tools.odds_ratio(exp3.trivial, exp3[fac] == f)
              for f in exp3[fac].unique()]
             for fac in factors]

# And running some simple linear regressions for accuracy
micro_mod2 = smf.ols('acc_diff ~ loss + goal + class_balance + group_balance',
                   data=exp2)
micro_res2 = micro_mod2.fit()
micro_mod3 = smf.ols('acc_diff ~ loss + goal + class_balance + group_balance',
                   data=exp3)
micro_res3 = micro_mod3.fit()
