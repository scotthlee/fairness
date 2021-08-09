import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools
import balancers
# NOTE: When comparing Scott's and Preston's code reshape and transpose Scott's code as follows:
#       y_der_scott.reshape((num_groups, num_classes, num_classes)).transpose(0, 2, 1) then print
#       y_der_preston.reshape((num_classes, num_classes, num_groups))[:, :, i] for different values of
#       i to compare. They should then be matched up properly if the two code bases are doing the same thing.

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

print('TESTING DEMOGRAPHIC PARITY')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='demographic_parity')


df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal
print('TESTING PYA WEIGHTED O-1 SHOULD BE SAME AS MACRO LOSS')
print('SCOTT\'s CODE WITH MACRO LOSS')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust(loss='macro', goal='odds')

print('PRESTON\'S WEIGHTED PYA 0-1 LOSS')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='pya_weighted_01', goal='odds')


mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='positive_predictive_parity')

