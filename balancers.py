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
                 y,
                 y_,
                 a,
                 summary=True,
                 threshold_objective='roc'):
        # Setting the targets
        self.y = y
        self.y_ = y_
        self.a = a
        self.rocs = None
        self.roc = None
        self.con = None
        self.goal = None
        self.thr_obj = threshold_objective
        
        # Getting the group info
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]
        self.p = [len(cols) / len(y) for cols in group_ids]
        
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
                self.y_ = probs.astype(np.uint8)
        
        # Calcuating the groupwise classification rates
        self._gr_list = [tools.CLFRates(self.y[i], self.y_[i]) 
                         for i in group_ids]
        self.group_rates = dict(zip(self.groups, self._gr_list))
        
        # And then the overall rates
        self.overall_rates = tools.CLFRates(self.y, self.y_)
        
        if summary:
            self.summary(adj=False)
        
        
    def adjust(self,
               goal='odds',
               round=4,
               return_optima=True,
               summary=True,
               binom=False):
        
        self.goal = goal
        
        # Getting the coefficients for the objective
        dr = [(g.nr * self.p[i], g.pr * self.p[i]) 
              for i, g in enumerate(self._gr_list)]
        
        # Getting the overall error rates and group proportions
        s = self.overall_rates.acc
        e = 1 - s
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(s - e) * r[0], 
                               (e - s) * r[1]]
                             for r in dr]).flatten()
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
        
        # Choosing whether to go with equalized odds or opportunity
        if 'odds' in goal:
            self.con = np.vstack((tprs, fprs))
        elif 'opportunity' in goal:
            self.con = tprs
        elif 'parity' in goal:
            pass
        
        con_b = np.zeros(self.con.shape[0])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=self.con,
                                       b_eq=con_b,
                                       method='highs')
        self.pya = self.opt.x.reshape(len(self.groups), 2)
        
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
        
        if summary:
            self.summary(org=False)
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
    
    def predict(self, y_, a, binom=False):
        adj = tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             preds=False,
             optimum=True,
             roc_curves=True,
             lp_lines=False, 
             chance_line=True,
             alpha=0.5):
        # Setting plot shape and figuring out if we're in ROC or PR space
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = tools.group_roc_coords(self.y, self.y_, self.a)
        plt.scatter(x=orig_coords.fpr,
                    y=orig_coords.tpr, 
                    color='red',
                    alpha=alpha)
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = tools.group_roc_coords(self.y, self.y_adj, self.a)
            plt.scatter(x=adj_coords.fpr, 
                        y=adj_coords.tpr, 
                        color='blue', 
                        alpha=alpha)
        
        # Optionally adding the ROC curves
        if self.rocs is not None and roc_curves:
            [plt.plot(r[0], r[1]) for r in self.rocs]
        
        # Optionally adding the chance line
        if chance_line:
            plt.plot((0, 1), (0, 1),
                     color='lightgray')
        
        # Adding lines to show the LP geometry
        if lp_lines:
            pass
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            err_mess1 = '.adjust() must be called '
            err_mess2 = 'before the optimum balance point can be shown.'
            assert self.roc is not None, err_mess1 + err_mess2
            
            if 'odds' in self.goal:
                plt.scatter(self.roc[0],
                            self.roc[1],
                            marker='x',
                            color='black')
            
            elif 'opportunity' in self.goal:
                plt.hlines(self.roc[1],
                           xmin=0,
                           xmax=1,
                           color='black',
                           linestyles='--',
                           linewidths=0.5)
            
            elif 'parity' in self.goal:
                pass
        
        plt.show()
    
    def summary(self, org=True, adj=True):
        if org:
            org_coords = tools.group_roc_coords(self.y, self.y_, self.a)
            org_loss = 1 - self.overall_rates.acc
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            adj_coords = tools.group_roc_coords(self.y, self.y_adj, self.a)
            adj_loss = 1 - tools.CLFRates(self.y, self.y_adj).acc
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %adj_loss)

