import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools


# Importing the data
df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal
y_p = tools.p_vec(y)

# Getting the probability matrices
pm = tools.ProbabilityMatrices(y, y_, a)

# Getting the pairwise constraints
tpr_cons, fpr_cons, norm_cons = tools.constraint_pairs(pm)

# Writing out some different loss functions; micro is what Hardt et al.
# use in the 2016 paper, but it forces everything to the majority class;
# 'macro' seems to work much better
macro = -1 * pm.cp_mats.flatten()
w_macro = np.tile(np.repeat(y_p, 3), 3) * macro
micro = pm.loss_weights

# Testing the LP under strict equality
norm_bounds = np.repeat(1, norm_cons.shape[0])
tpr_bounds = np.repeat(0, tpr_cons.shape[0])
all_cons = np.concatenate([tpr_cons, norm_cons])
all_cons_bounds = np.concatenate([tpr_bounds, norm_bounds])
tpr_eq_opt = linprog(c=macro,
                     A_eq=all_cons,
                     b_eq=all_cons_bounds,
                     bounds=[0, 1],
                     method='highs')
tpr_eq_mats = tools.pars_to_cpmat(tpr_eq_opt)
tpr_eq_rocs = tools.parmat_to_roc(tpr_eq_mats,
                                  pm.p_vecs,
                                  pm.cp_mats)

# And then relaxing the constraints a bit
tpr_ineq_bounds = np.repeat(.1, 6)
tpr_ineq_opt = linprog(c=macro,
                       A_ub=tpr_cons,
                       b_ub=tpr_ineq_bounds,
                       A_eq=norm_cons,
                       b_eq=norm_bounds,
                       bounds=[0, 1],
                       method='highs')
tpr_ineq_mats = tools.pars_to_cpmat(tpr_ineq_opt)
tpr_ineq_rocs = tools.parmat_to_roc(tpr_ineq_mats,
                                    pm.p_vecs,
                                    pm.cp_mats)

# Trying with equalized odds; first with the relaxed constraints
eo_bounds = np.repeat(.05, 12)
eo_con = np.concatenate([tpr_cons, fpr_cons])
eo_opt = linprog(c=macro,
                 A_ub=eo_con,
                 b_ub=eo_bounds,
                 A_eq=norm_cons,
                 b_eq=norm_bounds,
                 bounds=[0, 1],
                 method='highs')
eo_mats = tools.pars_to_cpmat(eo_opt)
eo_rocs = tools.parmat_to_roc(eo_mats,
                              pm.p_vecs,
                              pm.cp_mats)

# And now with the strict constraints
eo_eq_bounds = np.repeat(0, 12)
eo_eq_opt = linprog(c=macro,
                    A_eq=np.concatenate([eo_con, norm_cons]),
                    b_eq=np.concatenate([eo_eq_bounds, norm_bounds]),
                    bounds=[0, 1],
                    method='highs')
eo_eq_mats = tools.pars_to_cpmat(eo_eq_opt)
eo_eq_rocs = tools.parmat_to_roc(eo_eq_mats,
                              pm.p_vecs,
                              pm.cp_mats)
