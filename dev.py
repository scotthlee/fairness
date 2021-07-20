import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools


# Importing the data
df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal

# Getting the probability matrices
pm = tools.ProbabilityMatrices(y, y_, a)

# Getting the pairwise constraints
tpr_cons, fpr_cons, norm_cons = tools.constraint_pairs(pm)

# Writing out some different loss functions;
# I think these are basically correct, but I'm not sure
w_macro = -1 * np.array([pm.group_probs[i] * pm.cp_mats[i]
                    for i in range(pm.n_groups)]).flatten()
macro = -1 * pm.cp_mats.flatten()
micro = pm.loss_weights

# Testing the LP under strict equality
norm_bounds = np.repeat(1, norm_cons.shape[0])
tpr_bounds = np.repeat(0, tpr_cons.shape[0])
all_cons = np.concatenate([tpr_cons, norm_cons])
all_cons_bounds = np.concatenate([tpr_bounds, norm_bounds])
tpr_eq_opt = linprog(c=w_macro,
                     A_eq=all_cons,
                     b_eq=all_cons_bounds,
                     bounds=[0, 1],
                     method='highs')
tpr_eq_mats = tools.pars_to_cpmat(tpr_eq_opt)

# And then relaxing the constraints a bit
tpr_ineq_bounds = np.repeat(.1, 6)
tpr_ineq_opt = linprog(c=w_macro,
                       A_ub=tpr_cons,
                       b_ub=tpr_ineq_bounds,
                       A_eq=norm_cons,
                       b_eq=norm_bounds,
                       bounds=[0, 1],
                       method='highs')
tpr_ineq_mats = tools.pars_to_cpmat(tpr_ineq_opt)

# Trying with equalized odds
eo_bounds = np.repeat(.01, 12)
eo_con = np.concatenate([tpr_cons, fpr_cons])
eo_opt = linprog(c=macro,
                 A_ub=eo_con,
                 b_ub=eo_bounds,
                 A_eq=norm_cons,
                 b_eq=norm_bounds,
                 bounds=[0, 1],
                 method='highs')

eo_mats = tools.pars_to_cpmat(eo_opt)
