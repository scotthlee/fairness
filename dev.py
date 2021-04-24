import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_curve, hinge_loss
from sklearn.ensemble import RandomForestClassifier
from importlib import reload
from matplotlib import pyplot as plt
from itertools import combinations

import balancers as b
import tools


# Importing some test data
records = pd.read_csv('records.csv')

# Keeping only the 3 most common race groups for now
records = records[(records.race == 'Black / African American') |
                  (records.race == 'White') |
                  (records.race == 'Undisclosed')]

# Setting the variables for the joint distribution
pcr = records.pcr.values
cough = records.cough.values
fever = records.fever.values
taste = records.losstastesmell.values
race_bin = np.array(records.race == 'White', dtype=np.uint8)
race = records.race.values
cc = records.cc4.values
X = records.iloc[:, 3:18].values

# Fitting a toy model
rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X, pcr)
rf_probs = rf.oob_decision_function_[:, 1]

# Testin gthe balancer
pb = b.PredictionBalancer(pcr, cc, race)
pb.adjust()
pb.summary()
pb.plot()

