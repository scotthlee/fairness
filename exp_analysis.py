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
levels2 = [exp2[f].unique() for f in factors]
vars2 = [[f + '.' + l for l in levels2[i]] for i, f in enumerate(factors)]
triv_ors2 = [[tools.odds_ratio(exp2.trivial != 0, exp2[fac] == f)
              for f in exp2[fac].unique()]
             for fac in factors]
triv_ors2 = pd.DataFrame([tools.flatten(vars2),
                          tools.flatten(triv_ors2)]).transpose()
triv_ors2.columns = ['setup', 'OR']
triv_ors2.to_csv('data/triv_ors2.csv', index=False)

levels3 = [exp3[f].unique() for f in factors]
vars3 = [[f + '.' + l for l in levels3[i]] for i, f in enumerate(factors)]
triv_ors3 = [[tools.odds_ratio(exp3.trivial != 0, exp3[fac] == f)
              for f in exp3[fac].unique()]
             for fac in factors]
triv_ors3 = pd.DataFrame([tools.flatten(vars3),
                          tools.flatten(triv_ors3)]).transpose()
triv_ors3.columns = ['setup', 'OR']
triv_ors3.to_csv('data/triv_ors3.csv', index=False)

# And running some simple linear regressions for accuracy
micro_f = 'acc_diff ~ loss + goal + class_balance + group_balance'
micro_mod2 = smf.ols(micro_f, data=exp2)
micro_res2 = micro_mod2.fit()
micro_mod3 = smf.ols(micro_f, data=exp3)
micro_res3 = micro_mod3.fit()

mm_f = 'mean_mean_tpr_diff ~ loss + goal + class_balance + group_balance'
mm_tpr_mod2 = smf.ols(mm_f, data=exp2)
mm_tpr_res2 = mm_tpr_mod2.fit()
mm_tpr_mod3 = smf.ols(mm_f, data=exp3)
mm_tpr_res3 = mm_tpr_mod3.fit()

mx_f = 'max_mean_tpr_diff ~ loss + goal + class_balance + group_balance'
mx_tpr_mod2 = smf.ols(mx_f, data=exp2)
mx_tpr_res2 = mx_tpr_mod2.fit()
mx_tpr_mod3 = smf.ols(mx_f, data=exp3)
mx_tpr_res3 = mx_tpr_mod3.fit()

