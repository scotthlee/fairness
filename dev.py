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

# Trying the multiclass balancer
mb = balancers.MulticlassBalancer(y, y_, a)
mb.adjust(loss='macro')
mb.summary()

# Trying the binary balancer
b = balancers.BinaryBalancer(df.shear, df.shear_pred, a)
b.adjust()

# And trying the multiclass balancer on the binary data
mb_b = balancers.MulticlassBalancer(df.shear.values, 
                                    df.shear_pred.values, 
                                    a.values)
mb_b.adjust()