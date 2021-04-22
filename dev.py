import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from itertools import combinations


class CLFRates:
    def __init__(self,
                 y, 
                 y_,
                 round=4):
        # Doing a crosstab to save some time
        self.tab = pd.crosstab(y_, y)
        
        # Getting the basic counts
        tn = self.tab.iloc[0, 0]
        fn = self.tab.iloc[0, 1]
        fp = self.tab.iloc[1, 0]
        tp = self.tab.iloc[1, 1]
        
        # Calculating the rates
        self.pr = np.round((tp + fp) / len(y), round)
        self.nr = np.round((tn + fn) / len(y), round)
        self.tnr = np.round(tn / (tn + fp), round)
        self.tpr = np.round(tp / (tp + fn), round)
        self.fnr = np.round(fn / (fn + tp), round)
        self.fpr = np.round(fp / (fp + tn), round)
        self.acc = (tn + tp) / len(y)


def from_top(roc_point, round=4):
    d = np.sqrt(roc_point[0]**2 + (roc_point[1] - 1)**2)
    return d

def roc_coords(y, y_, round=4):
    # Getting hte counts
    tab = pd.crosstab(y_, y)
    tn = tab.iloc[0, 0]
    fn = tab.iloc[0, 1]
    fp = tab.iloc[1, 0]
    tp = tab.iloc[1, 1]
    
    # Calculating the rates
    tpr = np.round(tp / (tp + fn), round)
    fpr = np.round(fp / (fp + tn), round)
    
    return (fpr, tpr)


def group_roc_coords(y, y_, a, round=4):
    groups = np.unique(a)
    group_ids = [np.where(a ==g)[0] for g in groups]
    coords = [roc_coords(y[i], y_[i], round) for i in group_ids]
    fprs = [c[0] for c in coords]
    tprs = [c[1] for c in coords]
    out = pd.DataFrame([groups, fprs, tprs]).transpose()
    out.columns = ['group', 'fpr', 'tpr']
    return out


def pred_from_pya(y_, a, pya, binom=False):
    # Getting the groups and making the initially all-zero predictor
    groups = np.unique(a)
    out = np.zeros(y_.shape[0])
    
    for i, g in enumerate(groups):
        # Pulling the fitted switch probabilities for the group
        p = pya[i]
        
        # Indices in the group from which to choose swaps
        pos = np.where((a == g) & (y_ == 1))[0]
        neg = np.where((a == g) & (y_ == 0))[0]
        
        if not binom:
            # Randomly picking the positive predictions
            pos_samp = np.random.choice(a=pos, 
                                        size=int(p[1] * len(pos)), 
                                        replace=False)
            neg_samp = np.random.choice(a=neg, 
                                        size=int(p[0] * len(neg)),
                                        replace=False)
            samp = np.concatenate((pos_samp, neg_samp)).flatten()
            out[samp] = 1
        else:
            # Getting the 1s from a binomial draw for extra randomness 
            out[pos] = np.random.binomial(1, p[1], len(pos))
            out[neg] = np.random.binomial(1, p[0], len(neg))
    
    return out.astype(np.uint8)


class PredictionBalancer:
    def __init__(self):
        pass
        
    def fit(self,
            y,
            y_,
            a,
            round=4,
            return_optima=True,
            binom=False):
        
        # Setting the basic attributes
        self.y = y
        self.y_ = y_
        self.a = a
        self.groups = np.unique(a)
        
        # Getting the row numbers for each group
        group_ids = [np.where(a == g)[0] for g in self.groups]
        
        # Getting the proportion for each group
        self.p = [np.round(len(cols) / len(y), round) for cols in group_ids]
        
        # Calcuating the groupwise classification rates
        group_rates = [CLFRates(y[i], y_[i]) for i in group_ids]
        self.group_rates = dict(zip(self.groups, group_rates))
        dr = [(g.pr - g.nr) for g in group_rates]
        
        # Getting the overall error rates and group proportions
        self.overall_rates = CLFRates(y, y_)
        s = self.overall_rates.acc
        e = 1 - s
        p = self.p
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(e - s) * r, 
                               (s - e) * r]
                             for r in dr]).flatten()
        obj_bounds = [(0, 1)]
        
        # Generating the pairs for comparison
        n_groups = len(self.groups)
        group_combos = list(combinations(self.groups, 2))
        id_combos = list(combinations(range(n_groups), 2))
        
        # Pair drop to keep things full-rank with 3 or more groups
        if n_groups > 2:
            del group_combos[-1]
            del id_combos[-1]
        
        col_combos = np.array(id_combos) * 2
        n_pairs = len(group_combos)
        
        # Making empty matrices to hold the pairwise constraint coefficients
        tprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        fprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        
        # Filling in the constraint matrices
        for i, cols in enumerate(col_combos):
            # Fetching the group-specific rates
            gc = group_combos[i]
            g0 = self.group_rates[gc[0]]
            g1 = self.group_rates[gc[1]]
            
            # Filling in the group-specific coefficients
            tprs[i, cols[0]] = g0.fnr
            tprs[i, cols[0] + 1] = g0.tpr
            tprs[i, cols[1]] = -g1.fnr
            tprs[i, cols[1] + 1] = -g1.tpr
            
            fprs[i, cols[0]] = g0.tnr
            fprs[i, cols[0] + 1] = g0.fpr
            fprs[i, cols[1]] = -g1.tnr
            fprs[i, cols[1] + 1] = -g1.fpr
        
        roc_coefs = np.vstack((tprs, fprs))
        self.roc_coefs = roc_coefs
        roc_bounds = np.zeros(roc_coefs.shape[0])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=roc_coefs,
                                       b_eq=roc_bounds)
        pya = self.opt.x.reshape(len(self.groups), 2)
        self.pya = np.round(pya, round)
        
        # Setting the adjusted predictions
        self.y_adj = pred_from_pya(y_=self.y_, 
                                   a=self.a,
                                   pya=self.pya, 
                                   binom=binom)
        self.loss = 1 - CLFRates(self.y, self.y_adj).acc
        
        # Optionally returning the optimal ROC and loss
        p0, p1 = self.pya[0][0], self.pya[0][1]
        group = self.group_rates[self.groups[0]]
        fpr = (group.tnr * p0) + (group.fpr * p1)
        tpr = (group.fnr * p0) + (group.tpr * p1)
        self.roc = (np.round(fpr, round), np.round(tpr, round))
        
        if return_optima:                
            return {'loss': self.loss, 'roc': self.roc}
    
    def predict(self, y_, a, binom=False):
        adj = pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, add_lines=False, alpha=0.5):
        # Plotting the unadjusted ROC coordinates
        orig_coords = group_roc_coords(self.y, self.y_, self.a)
        plt.scatter(x=orig_coords.fpr,
                    y=orig_coords.tpr, 
                    color='red',
                    alpha=alpha)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        # Plotting the adjusted coordinates
        adj_coords = group_roc_coords(self.y, self.y_adj, self.a)
        plt.scatter(x=adj_coords.fpr, 
                    y=adj_coords.tpr, 
                    color='blue', 
                    alpha=alpha)
        
        # Adding lines to show the LP geometry
        if add_lines:
            pass
        plt.show()
    
    def summary(self):
        adj_coords = group_roc_coords(self.y, self.y_adj, self.a)
        adj_loss = 1 - CLFRates(self.y, self.y_adj).acc
        org_coords = group_roc_coords(self.y, self.y_, self.a)
        org_loss = 1 - self.overall_rates.acc
        
        print('\nPre-adjustment group rates were \n')
        print(org_coords)
        print('\nAnd loss was %.4f' %org_loss)
        print('\n \n')
        print('Post-adjustment group rates are \n')
        print(adj_coords)
        print('\nAnd loss is %.4f\n' %adj_loss)


class ProbabilityBalancer:
    def __init__(self):
        pass
    
    def fit(y, probs, a, return_optima=True):
        self.y = y
        self.probs = probs
        self.a = a
        self.groups = np.unique(a)
        
        # Getting the row numbers for each group
        group_ids = [np.where(a == g)[0] for g in self.groups]
        
        # Getting the proportion for each group
        self.p = [np.round(len(cols) / len(y), round) for cols in group_ids]
        
        # Calcuating the groupwise classification rates
        group_rates = [CLFRates(y[i], y_[i]) for i in group_ids]
        self.group_rates = dict(zip(self.groups, group_rates))
        self.overall_rates = CLFRates(y, y_)


# Importing some test data
records = pd.read_csv('records.csv')

# Keeping only the 3 most common race groups for now
records = records[(records.race == 'Black') |
                  (records.race == 'White') |
                  (records.race == 'Asian')]

# Setting the variables for the joint distribution
pcr = np.repeat(records.pcr_pos.values, 10)
cough = np.repeat(records.cough.values, 10)
fever = np.repeat(records.fever_chills.values, 10)
taste = np.repeat(records.tastesmell_combo.values, 10)
race_bin = np.repeat(np.array(records.race == 'White', 
                              dtype=np.uint8), 10)
race = np.repeat(records.race.values, 10)

# Testing the balancer
pb = PredictionBalancer()
pb.fit(pcr, taste, race)
pb.summary()
pb.plot()

