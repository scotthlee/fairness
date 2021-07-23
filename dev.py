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
mb.adjust(obj='macro')
