
import pandas as pd
import numpy as np

from scipy.optimize import linprog

import tools
import balancers

df = pd.read_csv('data/farm_animals.csv')
y = df.action.values
y_ = df.pred_action.values
a = df.animal

mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds', slack=0)

print('-------------------------With Small slack--------------------------------------')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds', slack=.1)

print('-------------------------With Large slack--------------------------------------')
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust_new(loss='0-1', goal='odds', slack=.5)

