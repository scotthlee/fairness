"""Debiasing binary predictions with linear programming.

Implementation based on work by Hardt, Srebro, & Price (2016):
https://arxiv.org/pdf/1610.02413.pdf
"""

import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sns

from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy
from sklearn.metrics import roc_curve

import tools


class BinaryBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,
                 data=None,
                 summary=True,
                 threshold_objective='j'):
        """Initializes an instance of a PredictionBalancer.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,) or str
            The true labels, either as a binary array or as a string \
            specifying a column in data. 
        
        y_ : array-like of shape (n_samples,) or str
            The predicted labels, either as an int (predictions) or float \
            (probabilities) array, or as a string specifying a column in data.
        
        a : array-like of shape (n_samples,) or str
            The protected attribute, either as an array, or as a string \
            specifying the column in data.
            
        data : pd.DataFrame instance, default None
            (Optional) DataFrame from which to pull y, y_, and a.
        
        summary : bool, default True
            Whether to print pre-adjustment false-positive and true-positive \
            rates for each group.
        
        threshold_objective : str, default 'j'
            Objective to use in evaluating thresholds when y_ contains \
            probabilities. Default is Youden's J index, or TPR - (1 - FPR) + 1.
        
        
        Attributes
        ----------
        actual_loss : float
            Loss on the current set of predictions in y_adj. 
        
        con : ndarray
            The coefficients of the constraint matrix for the linear program.
        
        goal : {'odds', 'opportunity'}, default 'odds'
            The fairness constraint to satisfy. Options are equalized odds \
            or equal opportunity. Set during .adjust().
        
        group_rates : dict
            The unadjusted tools.CLFRates object for each group.
        
        overall_rates : tools.CLFRates object
            The unadjusted CLFRates for the data overall.
        
        p : ndarray of shape (n_groups,)
            The proportions for each level of the protected attribute.
        
        pya : ndrray of shape (n_groups, 2)
            (P(y~=1 | y_=0), P(y~=1 | y_=1)) for each group after adjustment. \
            Set during .adjust().
        
        opt : SciPy.Optimize.OptimizeResult
            Optimizer solved by .adjust().
        
        rocs : list of sklearn.metrics.roc_curve results
            The roc curves for each group. Set only when y_ contains \
            probabilities.
        
        roc : tuple
            The theoretical optimum for (fpr, tpr) under equalized odds. Set \
            during .adjiust().
        
        theoretical_loss : float
            The theoretical optimum for loss given the constraints.
        
        y_adj : ndarray of shape (n_samples,)
            Predictions generated using the post-adjustment probabilities in \
            pya. Set on .adjust().
        """
        # Optional pull from a pd.DataFrame()
        if data is not None:
            y = data[y].values
            y_ = data[y_].values
            a = data[a].values
            
        # Setting the targets
        self.__y = y
        self.__y_ = y_
        self.__a = a
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
            self.__roc_stats = [tools.loss_from_roc(y[ids], 
                                                    probs[ids], 
                                                    self.rocs[i]) 
                              for i, ids in enumerate(group_ids)]
            if self.thr_obj == 'j':
                cut_ids = [np.argmax(rs['js']) for rs in self.__roc_stats]
                self.cuts = [self.rocs[i][2][id] for i, id in enumerate(cut_ids)]
                for g, cut in enumerate(self.cuts):
                    probs[group_ids[g]] = tools.threshold(probs[group_ids[g]],
                                                          cut)
                self.__y_ = probs.astype(np.uint8)
        
        # Calcuating the groupwise classification rates
        self.__gr_list = [tools.CLFRates(self.__y[i], self.__y_[i]) 
                         for i in group_ids]
        self.group_rates = dict(zip(self.groups, self.__gr_list))
        
        # And then the overall rates
        self.overall_rates = tools.CLFRates(self.__y, self.__y_)
        
        if summary:
            self.summary(adj=False)
        
        
    def adjust(self,
               goal='odds',
               round=4,
               return_optima=True,
               summary=True,
               binom=False):
        """Adjusts predictions to satisfy a fairness constraint.
        
        Parameters
        ----------
        goal : {'odds', 'opportunity'}, default 'odds'
            The constraint to be satisifed. Equalized odds and equal \
            opportunity are currently supported.
        
        round : int, default 4
            Decimal places for rounding results.
        
        return_optima: bool, default True
            Whether to reutn optimal loss and ROC coordinates.
        
        summary : bool, default True
            Whether to print post-adjustment false-positive and true-positive \
            rates for each group.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        (optional) optima : dict
            The optimal loss and ROC coordinates after adjustment.    
        """
        
        self.goal = goal
        
        # Getting the coefficients for the objective
        dr = [(g.nr * self.p[i], g.pr * self.p[i]) 
              for i, g in enumerate(self.__gr_list)]
        
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
        self.y_adj = tools.pred_from_pya(y_=self.__y_, 
                                         a=self.__a,
                                         pya=self.pya, 
                                         binom=binom)
        
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - tools.CLFRates(self.__y, self.y_adj).acc
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
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if np.any([0 < x < 1 for x in y_]):
            group_ids = [np.where(a == g)[0] for g in self.groups]
            y_ = deepcopy(y_)
            for g, cut in enumerate(self.cuts):
                y_[group_ids[g]] = tools.threshold(y_[group_ids[g]], cut)
        
        # Returning the adjusted predictions
        adj = tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             s1=50,
             s2=50,
             preds=False,
             optimum=True,
             roc_curves=True,
             lp_lines='all', 
             shade_hull=True,
             chance_line=True,
             palette='colorblind',
             style='white',
             xlim=(0, 1),
             ylim=(0, 1),
             alpha=0.5):
        """Generates a variety of plots for the PredictionBalancer.
        
        Parameters
        ----------
        s1, s2 : int, default 50
            The size parameters for the unadjusted (1) and adjusted (2) ROC \
            coordinates.
        
        preds : bool, default False
            Whether to observed ROC values for the adjusted predictions (as \
            opposed to the theoretical optima).
        
        optimum : bool, default True
            Whether to plot the theoretical optima for the predictions.
        
        roc_curves : bool, default True
            Whether to plot ROC curves for the unadjusted scores, when avail.
        
        lp_lines : {'upper', 'all'}, default 'all'
            Whether to plot the convex hulls solved by the linear program.
        
        shade_hull : bool, default True
            Whether to fill the convex hulls when the LP lines are shown.
        
        chance_line : bool, default True
            Whether to plot the line ((0, 0), (1, 1))
        
        palette : str, default 'colorblind'
            Color palette to pass to Seaborn.
        
        style : str, default 'dark'
            Style argument passed to sns.set_style()
        
        alpha : float, default 0.5
            Alpha parameter for scatterplots.
        
        Returns
        -------
        A plot showing shapes were specified by the arguments.
        """
        # Setting basic plot parameters
        plt.xlim(xlim)
        plt.ylim(ylim)
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = tools.group_roc_coords(self.__y, 
                                             self.__y_, 
                                             self.__a)
        sns.scatterplot(x=orig_coords.fpr,
                        y=orig_coords.tpr,
                        hue=self.groups,
                        s=s1,
                        palette='colorblind')
        plt.legend(loc='lower right')
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = tools.group_roc_coords(self.__y, 
                                                self.y_adj, 
                                                self.__a)
            sns.scatterplot(x=adj_coords.fpr, 
                            y=adj_coords.tpr,
                            hue=self.groups,
                            palette='colorblind',
                            marker='x',
                            legend=False,
                            s=s2,
                            alpha=1)
        
        # Optionally adding the ROC curves
        if self.rocs is not None and roc_curves:
            [plt.plot(r[0], r[1]) for r in self.rocs]
        
        # Optionally adding the chance line
        if chance_line:
            plt.plot((0, 1), (0, 1),
                     color='lightgray')
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting the groupwise coordinates
            group_rates = self.group_rates.values()
            group_var = np.array([[g]*3 for g in self.groups]).flatten()
            
            # Getting coordinates for the upper portions of the hulls
            upper_x = np.array([[0, g.fpr, 1] for g in group_rates]).flatten()
            upper_y = np.array([[0, g.tpr, 1] for g in group_rates]).flatten()
            upper_df = pd.DataFrame((upper_x, upper_y, group_var)).T
            upper_df.columns = ['x', 'y', 'group']
            upper_df = upper_df.astype({'x': 'float',
                                        'y': 'float',
                                        'group': 'str'})
            # Plotting the line
            sns.lineplot(x='x', 
                         y='y', 
                         hue='group', 
                         data=upper_df,
                         alpha=0.75, 
                         legend=False)
            
            # Optionally adding lower lines to complete the hulls
            if lp_lines == 'all':
                lower_x = np.array([[0, 1 - g.fpr, 1] 
                                    for g in group_rates]).flatten()
                lower_y = np.array([[0, 1 - g.tpr, 1] 
                                    for g in group_rates]).flatten()
                lower_df = pd.DataFrame((lower_x, lower_y, group_var)).T
                lower_df.columns = ['x', 'y', 'group']
                lower_df = lower_df.astype({'x': 'float',
                                            'y': 'float',
                                            'group': 'str'})
                # Plotting the line
                sns.lineplot(x='x', 
                             y='y', 
                             hue='group', 
                             data=lower_df,
                             alpha=0.75, 
                             legend=False)
                
            # Shading the area under the lines
            if shade_hull:
                for i, group in enumerate(self.groups):
                    uc = upper_df[upper_df.group == group]
                    u_null = np.array([0, uc.x.values[1], 1])
                                        
                    if lp_lines == 'upper':
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                    if lp_lines == 'all':
                        lc = lower_df[lower_df.group == group]
                        l_null = np.array([0, lc.x.values[1], 1])
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                        plt.fill_between(x=lc.x,
                                         y1=l_null,
                                         y2=lc.y,
                                         color=cmap[i],
                                         alpha=0.2)        
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.roc is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            elif 'odds' in self.goal:
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
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        if org:
            org_coords = tools.group_roc_coords(self.__y, self.__y_, self.__a)
            org_loss = 1 - self.overall_rates.acc
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            adj_coords = tools.group_roc_coords(self.__y, self.y_adj, self.__a)
            adj_loss = 1 - tools.CLFRates(self.__y, self.y_adj).acc
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %adj_loss)


class MulticlassBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,
                 data=None,
                 summary=False,
                 threshold_objective='j'):
        """Initializes an instance of a PredictionBalancer.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,) or str
            The true labels, either as an array or as a string \
            specifying a column in data. 
        
        y_ : array-like of shape (n_samples,) or str
            The predicted labels, either as an array or as a string \
            specifying a column in data.
        
        a : array-like of shape (n_samples,) or str
            The protected attribute, either as an array, or as a string \
            specifying the column in data.
            
        data : pd.DataFrame instance, default None
            (Optional) DataFrame from which to pull y, y_, and a.
        
        summary : bool, default True
            Whether to print pre-adjustment false-positive and true-positive \
            rates for each group.
        
        Attributes
        ----------
        actual_loss : float
            Loss on the current set of predictions in y_adj. 
        
        con : ndarray
            The coefficients of the constraint matrix for the linear program.
        
        goal : {'odds', 'opportunity'}, default 'odds'
            The fairness constraint to satisfy. Options are equalized odds \
            or equal opportunity. Set during .adjust().
        
        group_rates : dict
            The unadjusted tools.CLFRates object for each group.
        
        overall_rates : tools.CLFRates object
            The unadjusted CLFRates for the data overall.
        
        p : ndarray of shape (n_groups,)
            The proportions for each level of the protected attribute.
        
        pya : ndrray of shape (n_groups, 2)
            (P(y~=1 | y_=0), P(y~=1 | y_=1)) for each group after adjustment. \
            Set during .adjust().
        
        opt : SciPy.Optimize.OptimizeResult
            Optimizer solved by .adjust().
        
        rocs : list of sklearn.metrics.roc_curve results
            The roc curves for each group. Set only when y_ contains \
            probabilities.
        
        roc : tuple
            The theoretical optimum for (fpr, tpr) under equalized odds. Set \
            during .adjiust().
        
        theoretical_loss : float
            The theoretical optimum for loss given the constraints.
        
        y_adj : ndarray of shape (n_samples,)
            Predictions generated using the post-adjustment probabilities in \
            pya. Set on .adjust().
        """
        # Optional pull from a pd.DataFrame()
        if data is not None:
            y = data[y].values
            y_ = data[y_].values
            a = data[a].values
            
        # Setting the targets
        self.__y = y
        self.__y_ = y_
        self.__a = a
        self.rocs = None
        self.roc = None
        self.con = None
        self.goal = None
        
        # Getting the group info
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]
        self.p_y = tools.p_vec(y)
        self.p_a = tools.p_vec(a)
        self.n_classes = self.p_y.shape[0]
        
        # Getting some basic info for each group
        self.groups = np.unique(a)
        self.n_groups = len(self.groups)
        self.group_probs = tools.p_vec(a)
        group_ids = [np.where(a == g) for g in self.groups]
        
        # Getting the group-specific P(Y), P(Y- | Y), and constraint matrices
        p_vecs = np.array([tools.p_vec(y[ids]) for ids in group_ids])
        self.p_vecs = self.p_a.reshape(-1, 1) * p_vecs
        self.cp_mats = np.array([tools.cp_mat(y[ids], y_[ids]) 
                                 for ids in group_ids])
        
        if summary:
            pass
            self.summary(adj=False)
    
    def __get_constraints(self, p_vec, cp_mat):
        '''Calculates TPR and FPR weights for the constraint matrix'''
        # Shortening the vars to keep things clean
        p = p_vec
        M = cp_mat
        
        # Setting up the matrix of parameter weights
        n_classes = M.shape[0]
        n_params = n_classes**2
        tpr = np.zeros(shape=(n_classes, n_params))
        fpr = np.zeros(shape=(n_classes, n_params))
        off = np.zeros(shape=(n_classes, n_classes - 1, n_params))
        
        # Filling in the weights
        for i in range(n_classes):
            # Dropping row to calculate FPR
            p_i = np.delete(p, i)
            M_i = np.delete(M, i, 0)
            
            start = i * (n_classes)
            end = start + n_classes
            fpr[i, start:end] = np.dot(p_i, M_i) / p_i.sum()
            tpr[i, start:end] = M[i]
            
            for j in range(n_classes - 1):
                off[i, j, start:end] = M_i[j]
        
        # Reshaping the off-diagonal constraints
        off = np.concatenate(off, 0)
        
        return tpr, fpr, off
    
    def __pair_constraints(self, constraints):
        '''Takes the output of constraint_weights() and returns a matrix 
        of the pairwise constraints
        '''
        # Setting up the preliminaries
        tprs = np.array([c[0] for c in constraints])
        fprs = np.array([c[1] for c in constraints])
        off = np.array([c[2] for c in constraints])
        n_params = tprs.shape[2]
        n_classes = tprs.shape[1]
        n_groups = self.n_groups
        group_combos = list(combinations(range(n_groups), 2))[:-1]
        n_pairs = len(group_combos)
        
        # Setting up the empty matrices
        tpr_cons = np.zeros(shape=(n_pairs,
                                   n_groups,
                                   n_classes, 
                                   n_params))
        fpr_cons = np.zeros(shape=(n_pairs,
                                   n_groups,
                                   n_classes,
                                   n_params))
        off_cons = np.zeros(shape=(n_pairs,
                                   n_groups,
                                   n_classes * (n_classes - 1),
                                   n_params))
        
        # Filling in the constraint comparisons
        for i, c in enumerate(group_combos):
            # Getting the original diffs for flipping
            diffs = self.cp_mats[c[0]] - self.cp_mats[c[1]]
            tpr_flip = np.sign(np.diag(diffs)).reshape(-1, 1)
            fpr_flip = np.sign(np.sum(fprs[c[0]], 1) - np.sum(fprs[c[1]], 1))
            fpr_flip = fpr_flip.reshape(-1, 1)
            fpr_flip[np.where(fpr_flip == 0)] = 1
            
            # Filling in the constraints
            tpr_cons[i, c[0]] = tpr_flip * tprs[c[0]]
            tpr_cons[i, c[1]] = tpr_flip * -1 * tprs[c[1]]
            fpr_cons[i, c[0]] = fpr_flip * fprs[c[0]]
            fpr_cons[i, c[1]] = fpr_flip * -1 * fprs[c[1]]
            off_cons[i, c[0]] = off[c[0]]
            off_cons[i, c[1]] = off[c[1]]
        
        # Filling in the norm constraints
        one_cons = np.zeros(shape=(n_groups * n_classes,
                                   n_classes * n_params))
        cols = np.array(list(range(0, 
                                   n_groups * n_classes**2, 
                                   n_classes)))
        cols = cols.reshape(n_groups, n_classes)
        i = 0
        for c in cols:
            for j in range(n_classes):
                one_cons[i, c + j] = 1
                i += 1
        
        # Reshaping the arrays
        tpr_cons = np.concatenate([np.hstack(m) for m in tpr_cons])
        fpr_cons = np.concatenate([np.hstack(m) for m in fpr_cons])
        off_cons = np.concatenate([np.hstack(m) for m in off_cons])
        
        return tpr_cons, fpr_cons, off_cons, one_cons
    
    def adjust(self,
               goal='odds',
               loss='macro',
               round=4,
               return_optima=False,
               summary=False,
               binom=False):
        """Adjusts predictions to satisfy a fairness constraint.
        
        Parameters
        ----------
        goal : {'odds', 'opportunity'}, default 'odds'
            The constraint to be satisifed. Equalized odds and equal \
            opportunity are currently supported.
        
        loss : {'macro', 'w_macro', 'micro'}, default 'macro'
            The loss function to optimize.
        
        round : int, default 4
            Decimal places for rounding results.
        
        return_optima: bool, default True
            Whether to reutn optimal loss and ROC coordinates.
        
        summary : bool, default True
            Whether to print post-adjustment false-positive and true-positive \
            rates for each group.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        (optional) optima : dict
            The optimal loss and ROC coordinates after adjustment.    
        """
        # Getting the costraint weights
        constraints = [self.__get_constraints(self.p_vecs[i],
                                              self.cp_mats[i])
                               for i in range(self.n_groups)]
        self.constraints = constraints
        
        # Setting the objective for optimization
        if loss == 'macro':
            self.obj = -1 * self.cp_mats.flatten()
            '''
            # Alternative form as the sum of the off-diagonals
            off_loss = [[np.delete(a, i, 0).sum(0) 
                         for i in range(self.n_classes)]
                        for a in self.cp_mats]
            self.obj = np.array(off_loss).flatten()
            '''
        
        elif loss == 'w_macro':
            macro = -1 * self.cp_mats.flatten()
            self.obj = np.tile(np.repeat(self.p_y, 3), 3) * macro
        
        elif loss == 'micro':
            tprs = np.array([c[0] for c in constraints])
            tpr_sums = np.array([np.dot(self.p_vecs[i], tprs[i]) 
                        for i in range(self.n_groups)])
            self.obj = -1 * tpr_sums.flatten()
        
        # Arranging the constraint weights by group comparisons
        tpr_cons, fpr_cons, off_cons, norm_cons = self.__pair_constraints(
            constraints
            )
        
        # Normazliation bounds; used in all scenarios
        norm_bounds = np.repeat(1, norm_cons.shape[0])
        
        # Choosing whether to go with equalized odds or opportunity
        if 'odds' in goal:
            con = np.concatenate([tpr_cons, fpr_cons, norm_cons])
            eo_bounds = np.repeat(0, tpr_cons.shape[0] * 2)
            con_bounds = np.concatenate([eo_bounds, norm_bounds])
        
        elif 'opportunity' in goal:
            con = np.concatenate([tpr_cons, norm_cons])
            tpr_bounds = np.repeat(0, tpr_cons.shape[0])
            con_bounds = np.concatenate([tpr_bounds, norm_bounds])
        
        elif 'strict' in goal:
            con = np.concatenate([off_cons, norm_cons])
            off_bounds = np.repeat(0, off_cons.shape[0])
            con_bounds = np.concatenate([off_bounds, norm_bounds])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=self.obj,
                                       bounds=[0, 1],
                                       A_eq=con,
                                       b_eq=con_bounds,
                                       method='highs')
        
        # Getting the Y~ matrices
        self.m = tools.pars_to_cpmat(self.opt,
                                     n_groups=self.n_groups,
                                     n_classes=self.n_classes)
        
        # Calculating group-specific ROC scores from the new parameters
        self.rocs = tools.parmat_to_roc(self.m,
                                        self.p_vecs,
                                        self.cp_mats)
        
        if summary:
            self.summary(org=False)
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
    
    def predict(self, y_, a, binom=False):
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if np.any([0 < x < 1 for x in y_]):
            group_ids = [np.where(a == g)[0] for g in self.groups]
            y_ = deepcopy(y_)
            for g, cut in enumerate(self.cuts):
                y_[group_ids[g]] = tools.threshold(y_[group_ids[g]], cut)
        
        # Returning the adjusted predictions
        adj = tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             s1=50,
             s2=50,
             preds=False,
             optimum=True,
             roc_curves=True,
             lp_lines='all', 
             shade_hull=True,
             chance_line=True,
             palette='colorblind',
             style='white',
             xlim=(0, 1),
             ylim=(0, 1),
             alpha=0.5):
        """Generates a variety of plots for the PredictionBalancer.
        
        Parameters
        ----------
        s1, s2 : int, default 50
            The size parameters for the unadjusted (1) and adjusted (2) ROC \
            coordinates.
        
        preds : bool, default False
            Whether to observed ROC values for the adjusted predictions (as \
            opposed to the theoretical optima).
        
        optimum : bool, default True
            Whether to plot the theoretical optima for the predictions.
        
        roc_curves : bool, default True
            Whether to plot ROC curves for the unadjusted scores, when avail.
        
        lp_lines : {'upper', 'all'}, default 'all'
            Whether to plot the convex hulls solved by the linear program.
        
        shade_hull : bool, default True
            Whether to fill the convex hulls when the LP lines are shown.
        
        chance_line : bool, default True
            Whether to plot the line ((0, 0), (1, 1))
        
        palette : str, default 'colorblind'
            Color palette to pass to Seaborn.
        
        style : str, default 'dark'
            Style argument passed to sns.set_style()
        
        alpha : float, default 0.5
            Alpha parameter for scatterplots.
        
        Returns
        -------
        A plot showing shapes were specified by the arguments.
        """
        # Setting basic plot parameters
        plt.xlim(xlim)
        plt.ylim(ylim)
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = tools.group_roc_coords(self.__y, 
                                             self.__y_, 
                                             self.__a)
        sns.scatterplot(x=orig_coords.fpr,
                        y=orig_coords.tpr,
                        hue=self.groups,
                        s=s1,
                        palette='colorblind')
        plt.legend(loc='lower right')
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = tools.group_roc_coords(self.__y, 
                                                self.y_adj, 
                                                self.__a)
            sns.scatterplot(x=adj_coords.fpr, 
                            y=adj_coords.tpr,
                            hue=self.groups,
                            palette='colorblind',
                            marker='x',
                            legend=False,
                            s=s2,
                            alpha=1)
        
        # Optionally adding the ROC curves
        if self.rocs is not None and roc_curves:
            [plt.plot(r[0], r[1]) for r in self.rocs]
        
        # Optionally adding the chance line
        if chance_line:
            plt.plot((0, 1), (0, 1),
                     color='lightgray')
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting the groupwise coordinates
            group_rates = self.group_rates.values()
            group_var = np.array([[g]*3 for g in self.groups]).flatten()
            
            # Getting coordinates for the upper portions of the hulls
            upper_x = np.array([[0, g.fpr, 1] for g in group_rates]).flatten()
            upper_y = np.array([[0, g.tpr, 1] for g in group_rates]).flatten()
            upper_df = pd.DataFrame((upper_x, upper_y, group_var)).T
            upper_df.columns = ['x', 'y', 'group']
            upper_df = upper_df.astype({'x': 'float',
                                        'y': 'float',
                                        'group': 'str'})
            # Plotting the line
            sns.lineplot(x='x', 
                         y='y', 
                         hue='group', 
                         data=upper_df,
                         alpha=0.75, 
                         legend=False)
            
            # Optionally adding lower lines to complete the hulls
            if lp_lines == 'all':
                lower_x = np.array([[0, 1 - g.fpr, 1] 
                                    for g in group_rates]).flatten()
                lower_y = np.array([[0, 1 - g.tpr, 1] 
                                    for g in group_rates]).flatten()
                lower_df = pd.DataFrame((lower_x, lower_y, group_var)).T
                lower_df.columns = ['x', 'y', 'group']
                lower_df = lower_df.astype({'x': 'float',
                                            'y': 'float',
                                            'group': 'str'})
                # Plotting the line
                sns.lineplot(x='x', 
                             y='y', 
                             hue='group', 
                             data=lower_df,
                             alpha=0.75, 
                             legend=False)
                
            # Shading the area under the lines
            if shade_hull:
                for i, group in enumerate(self.groups):
                    uc = upper_df[upper_df.group == group]
                    u_null = np.array([0, uc.x.values[1], 1])
                                        
                    if lp_lines == 'upper':
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                    if lp_lines == 'all':
                        lc = lower_df[lower_df.group == group]
                        l_null = np.array([0, lc.x.values[1], 1])
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                        plt.fill_between(x=lc.x,
                                         y1=l_null,
                                         y2=lc.y,
                                         color=cmap[i],
                                         alpha=0.2)        
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.roc is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            elif 'odds' in self.goal:
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
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        if org:
            org_coords = tools.group_roc_coords(self.__y, self.__y_, self.__a)
            org_loss = 1 - self.overall_rates.acc
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            adj_coords = tools.group_roc_coords(self.__y, self.y_adj, self.__a)
            adj_loss = 1 - tools.CLFRates(self.__y, self.y_adj).acc
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %adj_loss)

