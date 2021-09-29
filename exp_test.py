import numpy as np
import pandas as pd

from balancers import MulticlassBalancer
import tools

y_test = tools.simulate_y([1, 2, 3], 
                          ['a', 'b', 'c'],
                          [.3, .7], 
                          [[.1, .1, .8], 
                           [.3, .5, .2]],
                          n=10000)
p_y_a = np.array([[[.1, .2, .7], 
                   [.6, .3, .1], 
                   [.1, .1, .8]],
                  [[.8, .1, .1], 
                   [.2, .5, .3], 
                   [.2, .2, .6]]])
yh_test = tools.simulate_yh(test_df=y_test, 
                            p_y_a=p_y_a, 
                            outcomes=[1, 2, 3])


group_ids = [np.where(yh_test.group == g)[0] 
             for g in yh_test.group.unique()]
cp_mats = [tools.cp_mat(yh_test.y[ids], yh_test.y_hat[ids])
           for ids in group_ids]