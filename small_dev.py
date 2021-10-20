
import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools
import balancers

df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal

print('-------------------------Farm Animals MOOOO------------------')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds', slack=0)



df = pd.read_csv('data/test_data.csv')
_, y = np.unique(df.y.values, return_inverse=True)
_, y_ = np.unique(df.y_hat.values, return_inverse=True)
_, a = np.unique(df.group.values, return_inverse=True)
print('-------------------------Test Data HISSS------------------')
print('SCOTTS')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust(loss='micro', goal='odds')
print('PRESTONS')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds', slack=0)

#mb = balancers.MulticlassBalancer(y, y_, a)
#mb.adjust_new(loss='0-1', goal='odds', slack=0)
#
#print('-------------------------With Small slack--------------------------------------')
#mb = balancers.MulticlassBalancer(y, y_, a)
#mb.adjust_new(loss='0-1', goal='odds', slack=.1)
#
#print('-------------------------With Large slack--------------------------------------')
#mb = balancers.MulticlassBalancer(y, y_, a)
#mb.adjust_new(loss='0-1', goal='odds', slack=.5)

