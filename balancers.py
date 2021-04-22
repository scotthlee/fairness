import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sbn

from matplotlib import pyplot as plt
from itertools import combinations

from tools import group_roc_coords, pred_from_pya, CLFRates


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
        #dr = [(g.pr - g.nr)*self.p[i] for i, g in enumerate(group_rates)]
        dr = [(g.nr*self.p[i], g.pr*self.p[i]) 
              for i, g in enumerate(group_rates)]
        
        # Getting the overall error rates and group proportions
        self.overall_rates = CLFRates(y, y_)
        s = self.overall_rates.acc
        e = 1 - s
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(s - e) * r[0], 
                               (e - s) * r[1]]
                             for r in dr]).flatten()
        print(obj_coefs)
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
        
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - CLFRates(self.y, self.y_adj).acc
        cmin = self.opt.fun
        tl = cmin + (e*self.overall_rates.nr) + (s*self.overall_rates.pr)
        self.theoretical_loss = tl
        
        # Calculating the theoretical balance point in ROC space
        p0, p1 = self.pya[0][0], self.pya[0][1]
        group = self.group_rates[self.groups[0]]
        fpr = (group.tnr * p0) + (group.fpr * p1)
        tpr = (group.fnr * p0) + (group.tpr * p1)
        self.roc = (np.round(fpr, round), np.round(tpr, round))
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
    
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
