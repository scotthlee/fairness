import numpy as np
import pandas as pd

from balancers import MulticlassBalancer
import tools


def test_run(outcomes,
             groups,
             p_group,
             p_y_group,
             p_yh_group,
             loss='micro',
             goal='odds'):
    # Simulating the input data
    y_test = tools.simulate_y(outcomes,
                              groups,
                              p_group,
                              p_y_group)
    yh_test = tools.simulate_yh(y_test,
                                p_yh_group,
                                outcomes)
    
    # Setting up the variables
    y = y_test.y.values
    a = y_test.group.values
    yh = y_test.y_hat.values
    
    # Running the optimizations
    b = MulticlassBalancer(y, yh, a)
    b.adjust(loss=loss, goal=goal)
    accuracy = b.opt.fun
    status = b.opt.status
    roc = b.rocs[0]
    
    # Bundling things up
    out = {
           'goal': goal,
           'loss': loss,
           'status': status,
           'accuracy': accuracy,
           'roc': roc
    }
    
    return out

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
