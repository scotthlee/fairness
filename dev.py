import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools
import balancers


# Importing the data
df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal

# Trying the balancer
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust(loss='micro')

#print('WITH BINARY BALANCER CODE')
#mb = balancers.BinaryBalancer(y, y_, a)
#mb.adjust(loss='micro')

# Trying the Updated Version, should be the same
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')


# Checking results in the 2d case
print('CHECKING RESULTS WITH BINARY LABEL OF MULTICLASS BALANCER')
y = np.array(df.action == 'shear', dtype=np.uint8)
y_ = np.array(df.pred_action == 'shear', dtype=np.uint8)
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')

print('SCOTT\'S BINARY CODE')
y = np.array(df.action == 'shear', dtype=np.uint8)
y_ = np.array(df.pred_action == 'shear', dtype=np.uint8)
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')

# Dummy data I can hand compute over
print('SMALL HAND-COMPUTABLE EXAMPLE WITH NO CLASS IMBALANCE ACROSS GROUPS')
a = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
y_ = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1])
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')

print('WITH MODERATE CLASS IMBALANCE ACROSS GROUPS')
a = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
y_ = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1])
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')

print('WITH SEVERE CLASS IMBALANCE ACROSS GROUPS')
a = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
y_ = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1])
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='opportunity')

print('WITH TWO MINORITY GROUPS, THREE GROUPS TOTAL')
a = np.array([0]*6 + [1]*6 + [2]*20)
print(a.shape)
y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1] + [0] * 10 + [1] * 10)
y_ = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds')
#mb = balancers.MulticlassBalancer(y, y_, a)
#mb.adjust(loss='micro')

