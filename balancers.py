import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sbn

from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy
from sklearn.metrics import roc_curve

import tools


class PredictionBalancer:
    def __init__(self,
                 lp_objective='accuracy',
                 threshold_objective='roc'):
        self.lp_obj = lp_objective
        self.thr_obj = threshold_objective
        self.rocs = None
        
    def fit(self,
            y,
            y_,
            a,
            round=4,
            return_optima=True,
            binom=False):
        
        # Getting the group info
        self.a = a
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]
        self.p = [np.round(len(cols) / len(y), round) for cols in group_ids]
        
        # Optionally thresholding probabilities to get class predictions
        if np.any([0 < x < 1 for x in y_]):
            print('Probabilities detected.\n')
            probs = deepcopy(y_)
            self.rocs = [roc_curve(y[ids], probs[ids]) for ids in group_ids]
            self.roc_stats = [tools.loss_from_roc(y[ids], probs[ids], self.rocs[i]) 
                              for i, ids in enumerate(group_ids)]
            if self.thr_obj == 'roc':
                cut_ids = [np.argmax(rs['js']) for rs in self.roc_stats]
                cuts = [self.rocs[i][2][id] for i, id in enumerate(cut_ids)]
                for g, cut in enumerate(cuts):
                    probs[group_ids[g]] = tools.threshold(probs[group_ids[g]],
                                                          cut)
                y_ = probs.astype(np.uint8)
        
        # Setting the basic attributes
        self.y = y
        self.y_ = y_
        
        # Calcuating the groupwise classification rates
        group_rates = [tools.CLFRates(y[i], y_[i]) for i in group_ids]
        self.group_rates = dict(zip(self.groups, group_rates))
        dr = [(g.nr*self.p[i], g.pr*self.p[i]) 
              for i, g in enumerate(group_rates)]
        
        # Getting the overall error rates and group proportions
        self.overall_rates = tools.CLFRates(y, y_)
        s = self.overall_rates.acc
        e = 1 - s
        
        # Setting up the coefficients for the objective function
        if self.lp_obj == 'accuracy':
            obj_coefs = np.array([[(s - e) * r[0], 
                                   (e - s) * r[1]]
                                 for r in dr]).flatten()
        elif self.objective == 'roc':
            pass
        
        obj_bounds = [(0, 1)]
        
        # Generating the pairs for comparison
        n_groups = len(self.groups)
        group_combos = list(combinations(self.groups, 2))
        id_combos = list(combinations(range(n_groups), 2))
        
        # Pair drop to keep things full-rank with 3 or more groups
        if n_groups > 2:
            n_comp = n_groups - 1
            group_combos = group_combos[:n_comp]
            id_combos = id_combos[:n_comp]
        
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
        self.y_adj = tools.pred_from_pya(y_=self.y_, 
                                   a=self.a,
                                   pya=self.pya, 
                                   binom=binom)
        
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - tools.CLFRates(self.y, self.y_adj).acc
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
        adj = tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             draw_chance=True,
             add_optimum=True,
             add_lines=False, 
             alpha=0.5):
        # Plotting the unadjusted ROC coordinates
        orig_coords = tools.group_roc_coords(self.y, self.y_, self.a)
        plt.scatter(x=orig_coords.fpr,
                    y=orig_coords.tpr, 
                    color='red',
                    alpha=alpha)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        # Plotting the adjusted coordinates
        adj_coords = tools.group_roc_coords(self.y, self.y_adj, self.a)
        plt.scatter(x=adj_coords.fpr, 
                    y=adj_coords.tpr, 
                    color='blue', 
                    alpha=alpha)
        
        # Optionally adding the ROC curves
        if self.rocs is not None:
            [plt.plot(r[0], r[1]) for r in self.rocs]
        
        # Optionally adding the chance line
        if draw_chance:
            plt.plot((0, 1), (0, 1),
                     color='lightgray')
        # Adding lines to show the LP geometry
        if add_lines:
            pass
        
        # Optionally adding the post-adjustment optimum
        if add_optimum:
            plt.scatter(self.roc[0],
                        self.roc[1],
                        marker='x',
                        color='black')
        
        plt.show()
    
    def summary(self):
        adj_coords = tools.group_roc_coords(self.y, self.y_adj, self.a)
        adj_loss = 1 - tools.CLFRates(self.y, self.y_adj).acc
        org_coords = tools.group_roc_coords(self.y, self.y_, self.a)
        org_loss = 1 - self.overall_rates.acc
        
        print('\nPre-adjustment group rates were \n')
        print(org_coords)
        print('\nAnd loss was %.4f' %org_loss)
        print('\n \n')
        print('Post-adjustment group rates are \n')
        print(adj_coords)
        print('\nAnd loss is %.4f\n' %adj_loss)
