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
               return_optima=False,
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
        self.goal = ''
        
        # Getting the group info
        self.p_y = tools.p_vec(y)
        self.p_a = tools.p_vec(a)
        self.n_classes = self.p_y.shape[0]
        self.outcomes = np.unique(y)
        
        # Getting some basic info for each group
        self.groups = np.unique(a)
        self.n_groups = len(self.groups)
        self.group_probs = tools.p_vec(a)
        group_ids = [np.where(a == g) for g in self.groups]
        
        # Getting the group-specific P(Y), P(Y- | Y), and constraint matrices
        p_vecs = np.array([tools.p_vec(y[ids]) for ids in group_ids])
        p_pred_vecs = np.array([tools.p_vec(y_[ids]) for ids in group_ids])
        self.p_pred_vecs = self.p_a.reshape(-1, 1) * p_pred_vecs
        self.p_vecs = self.p_a.reshape(-1, 1) * p_vecs
        self.cp_mats = np.array([tools.cp_mat(y[ids], y_[ids]) 
                                 for ids in group_ids])
        self.cp_mats_t = np.zeros(
                (self.n_classes, self.n_classes, self.n_groups)
        )
        for a in range(self.n_groups):
            self.cp_mats_t[:, :, a] = self.cp_mats[a].transpose()

        self.cp_mat = tools.cp_mat(y, y_)
        old_rocs = [tools.cpmat_to_roc(self.p_vecs[i],
                                       self.cp_mats[i])
                    for i in range(self.n_groups)]
        self.old_rocs = np.array(old_rocs)
        
        if summary:
            self.summary(adj=False)
    
    def __get_constraints(self, p_vec, p_a, cp_mat):
        '''Calculates TPR and FPR weights for the constraint matrix'''
        # Shortening the vars to keep things clean
        p = p_vec
        M = cp_mat
        
        # Setting up the matrix of parameter weights
        n_classes = M.shape[0]
        n_params = n_classes**2
        tpr = np.zeros(shape=(n_classes, n_params))
        fpr = np.zeros(shape=(n_classes, n_params))
        
        # Filling in the weights
        for i in range(n_classes):
            # Dropping row to calculate FPR
            p_i = np.delete(p, i)
            M_i = np.delete(M, i, 0)
            
            start = i * (n_classes)
            end = start + n_classes
            fpr[i, start:end] = np.dot(p_i, M_i) / p_i.sum()
            tpr[i, start:end] = M[i]
        
        # Reshaping the off-diagonal constraints
        strict = np.zeros(shape=(n_params, n_params))
        #A = np.array(p.T / p_a).T
        #B = np.array(M.T * A).T
        B = M
        for i in range(n_classes):
            start = i * n_classes
            end = start + n_classes
            strict[start:end, start:end] = B
        
        return tpr, fpr, strict
    
    def __pair_constraints(self, constraints):
        '''Takes the output of constraint_weights() and returns a matrix 
        of the pairwise constraints
        '''
        # Setting up the preliminaries
        tprs = np.array([c[0] for c in constraints])
        fprs = np.array([c[1] for c in constraints])
        strict = np.array([c[2] for c in constraints])
        n_params = tprs.shape[2]
        n_classes = tprs.shape[1]
        n_groups = self.n_groups
        if n_groups > 2:
            group_combos = list(combinations(range(n_groups), 2))[:-1]
        else:
            group_combos = [(0, 1)]
            
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
        strict_cons = np.zeros(shape=(n_pairs,
                                      n_groups,
                                      n_params,
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
            strict_cons[i, c[0]] = strict[c[0]]
            strict_cons[i, c[1]] = -1 * strict[c[1]]
        
        # Filling in the norm constraints
        one_cons = np.zeros(shape=(n_groups * n_classes,
                                   n_groups * n_params))
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
        strict_cons = np.concatenate([np.hstack(m) for m in strict_cons])
        
        return tpr_cons, fpr_cons, strict_cons, one_cons

    def adjust_new(self,
               goal='odds',
               loss='0-1',
               round=4,
               return_optima=False,
               summary=False,
               binom=False):
        """Adjusts predictions to satisfy a fairness constraint.
        
        Parameters
        ----------
        goal : {'odds', 'opportunity', 'strict'}, default 'odds'
            The constraint to be satisifed. Equalized odds sets the TPR and FPR rates
            equal. Opportunity only sets TPR to be equal. Strict sets all off-diagonal
            entries of the confusion matrix to be equal, and may not create a feasible
            linear program.
        
        loss : {'0-1'} default '0-1'
            The loss function to optimize in expectation
        
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
        # note: self.p_vecs contains the joint group probabilities of y and a
        #       self.cp_mats contains the probabilities of y^ given y and a
        #       self.cp_mat contains the probabilities of y^ given y
        # the above should be added to the attributes doc string

        if loss == '0-1':
            loss_coefs = self.get_0_1_loss_coefs()
        elif loss == 'pya_weighted_01':
            loss_coefs = self.get_pya_weighted_01_loss_coefs()
        else:
            raise ValueError('Loss type %s not recognized' %loss)

        # Create a n_constraints by
        # n_classes (y~) by n_classes (y^) by n_groups matrix of
        # constraints, adding in additional constraints as needed 
        # to the first dim
        
        if goal == 'odds':
            cons_mat = self.get_equal_odds_constraints()
        elif goal == 'opportunity':
            cons_mat = self.get_equal_opp_constraints()
        elif goal == 'strict':
            cons_mat = self.get_strict_constraints()
        elif goal == 'demographic_parity':
            cons_mat = self.get_demographic_parity_constraints()
        elif goal == 'positive_predictive_parity':
            cons_mat = self.get_pred_parity_constraints(type_='positive')
        elif goal == 'strict_predictive_parity':
            cons_mat = self.get_pred_parity_constraints(type_='strict')
        else:
            raise ValueError('Fairness type/goal %s not recognized' %goal)

        # Add in constraint for derived probabilities to sum to one
        # for every fixed group and (true) class they should be
        # normalized, so one constraint for every (group, class) pair
        cons_norm = np.zeros(
            (self.n_groups * self.n_classes,
            self.n_classes, self.n_classes, self.n_groups)
        )
        for con_idx in range(cons_norm.shape[0]):
            c = con_idx % int(self.n_classes)
            g = con_idx // int(self.n_classes)
            cons_norm[con_idx, c, :, g] = np.ones(self.n_classes)
        cons_norm = cons_norm.reshape(cons_norm.shape[0], -1)
        cons_mat = np.concatenate([cons_mat, cons_norm], axis=0)            
        #print(cons_mat)
            

        # Form the bounds of the constraints:
        #    For odds/opp/strict these are 0 since they are equalities.
        #    For the normalization constraints these are 1
        cons_bounds = np.zeros(cons_mat.shape[0])
        cons_bounds[-cons_norm.shape[0]:] = 1

        self.opt = sp.optimize.linprog(c=loss_coefs,
                                       bounds=[0, 1],
                                       A_eq=cons_mat,
                                       b_eq=cons_bounds,
                                       method='highs')
  
        print(self.opt)
        y_derived = self.opt.x.reshape([self.n_classes, self.n_classes, self.n_groups])
        self.m = y_derived
        print('checking diagonal of W with cp_mats_t')
        print([self.cp_mats_t[:, i, 0] @ self.m[:, i, 0] for i in range(self.n_classes)])
        print([self.cp_mats_t[:, i, 1] @ self.m[:, i, 1] for i in range(self.n_classes)])
#        print([self.cp_mats_t[:, i, 2] @ self.m[:, i, 2] for i in range(self.n_classes)])

        print('checking diagonal of W with cp_mats')
        print(self.cp_mats[0, 0, :] @ self.m[:, 0, 0])
        print(self.cp_mats[1, 0, :] @ self.m[:, 0, 1])

        print('--------------Learned derived predictions-------------')
        print(self.m[:, :, 0])
        print(self.m[:, :, 1])
 #       print(self.m[:, :, 2])
        print('checking the W matrix')
        W = np.einsum('ijk, jlk->ilk', self.cp_mats_t.transpose((1, 0, 2)), self.m)
        print(W[:, :, 0])
        print(W[:, :, 1])
  #      print(W[:, :, 2])
        self.rocs = tools.parmat_to_roc(y_derived, self.p_vecs, self.cp_mats)

        self.con = cons_mat
        self.con_bounds = cons_bounds

        if summary:
            self.summary(org=False)
            
        if return_optima:
            return {'loss': self.theoretical_loss, 'roc': self.roc}

    def get_0_1_loss_coefs(self):
        coefs = np.zeros((self.n_classes, self.n_classes, self.n_groups))
        for c in range(self.n_classes):
            # Form |C| X |A| vector of joint probabilities of y and a
            # for summation over the mismatches between derived y and
            # true labels
            pred_mismatch = np.ones((self.n_classes, 1))
            pred_mismatch[c] = 0
            p = self.p_vecs.transpose() * pred_mismatch
            
            # matrix product across first two dimensions
            coefs[:, c, :] = np.einsum('ijk,jk->ik', self.cp_mats_t, p)
        return coefs.flatten()

    def get_pya_weighted_01_loss_coefs(self):
        return -self.cp_mats_t.flatten()


    def get_equal_odds_constraints(self):

        # equal opportunity constrains TPR to be equal
        tpr_cons = self.get_equal_opp_constraints()

        # constrain FPR to be equal
        fpr_cons = self.get_fpr_cons()

        equal_odds_cons = np.concatenate([tpr_cons, fpr_cons], axis=0)
        return equal_odds_cons
 
    def get_fpr_cons(self):
        # form intermediate matrix
        V = np.zeros((self.n_classes, self.n_classes, self.n_groups))
        for c in range(self.n_classes):
            # Form |C| X |A| vector of joint probabilities of y and a
            # for summation over the mismatches between derived y and
            # true labels
            pred_mismatch = np.ones((self.n_classes, 1))
            pred_mismatch[c] = 0
            p_mis = self.p_vecs.transpose() * pred_mismatch
            
            # matrix product across first two dimensions
            temp = np.einsum('ijk,jk->ik', self.cp_mats_t, p_mis)
            V[:, c, :] = temp/np.sum(p_mis, axis=0)
        full_cons = self.get_equal_cons_given_mat(np.transpose(V, axes=[1, 0, 2]))
        # only need diagonals to be equal, so select diagonals
        fp_cons = np.einsum('ijjklm -> ijklm', full_cons)
        return fp_cons.reshape((self.n_groups - 1) * self.n_classes, -1)


    def get_equal_opp_constraints(self):
        full_cons = self.get_equal_cons_given_mat(np.transpose(self.cp_mats_t, axes=(1, 0, 2)))
        # only need diagonals to be equal, so select diagonals
        tp_cons = np.einsum('ijjklm -> ijklm', full_cons)
        return tp_cons.reshape((self.n_groups - 1) * self.n_classes, -1)

    def get_strict_constraints(self):
        cons = self.get_equal_cons_given_mat(np.transpose(self.cp_mats_t, axes=(1, 0, 2)))
        return cons.reshape((self.n_groups - 1) * self.n_classes**2, -1)

    def get_demographic_parity_constraints(self):
        p_pred_vecs = self.p_pred_vecs.transpose() # to have shape |C| X |A|
        J = p_pred_vecs/(np.sum(p_pred_vecs, axis=0))
        J = J.reshape((self.n_classes, 1, self.n_groups))
        # repeat so it works with get_equal_cons_given_mat
        J = np.ones((self.n_classes, self.n_classes, self.n_groups)) * J
        cons = self.get_equal_cons_given_mat(J.transpose(1, 0, 2))
        cons = cons[:, 0, :] # all rows will turn out the same
        return cons.reshape((self.n_groups - 1) * self.n_classes, -1)

    def get_pred_parity_constraints(self, type_='strict'):
        # Constrain numerator of predictive rates to be equal
        M = self.cp_mats_t *\
            self.p_vecs.transpose().reshape([self.n_classes, 1, 1, self.n_groups])
        cons_numer = self.get_equal_cons_given_mat(M.transpose(0, 2, 1, 3))
        if type_ == 'positive':
            # take diagonal only
            cons_numer = np.einsum('ijjklm -> ijklm', cons_numer)
            cons_numer = cons_numer.reshape((self.n_groups - 1) * self.n_classes, -1)
        elif type_ == 'strict':
            cons_numer = cons_numer.reshape((self.n_groups - 1) * self.n_classes**2, -1) 
        else:
            raise ValueError('Constraint type %s not recognized in get_pred_parity_constraints' %type_)


        # Constrain Demoniminator of predictive rates to be equal
        p_pred_vecs = self.p_pred_vecs.transpose() # to have shape |C| X |A|
        E = p_pred_vecs.reshape((self.n_classes, 1, self.n_groups))
        # repeat so it works with get_equal_cons_given_mat
        E = np.ones((self.n_classes, self.n_classes, self.n_groups)) * E
        cons_den = self.get_equal_cons_given_mat(E.transpose(1, 0, 2))

        cons_den = cons_den[:, 0, :] # all rows will turn out the same
        num_cons_den = (self.n_groups - 1) * self.n_classes
        cons_den = cons_den.reshape(num_cons_den, -1)

        return np.vstack([cons_numer, cons_den])
        
        
        

    def get_equal_cons_given_mat(self, M):
        '''Given some matrix A of values desired to be fair across
        protected attributes, and which can be decomposed as A = M * P^T
        where P^T are the derived probabilities given the predicted label, 
        compute the constraints needed to enforce fairness for all values of
        A across groups. Assumes P is shaped y_derived by y_pred by group.

        Parameters
        ----------
        M: Three or four dimensional np array in the decomposition A = M * P^T. 
            Must have shape (n_classes, n_classes, n_groups) or shape
            (n_classes, n_classes, n_classes, n_groups)

        Returns
        -------
        The constraints as a np array with shape:
            (n_groups - 1, n_classes, n_classes, n_classes, n_classes, n_groups)
            organized as follows:
                dim 0 - Represents setting A equal across groups in pairs group i
                    and i + 1. So entry 0 along this dim sets A equal across group 0
                    and 1. Entry 1 sets A equal across group 1 and group 2 etc.
                dim 1 and 2 - Represent setting each entry within A (A_{ij}) equal 
                    across groups for a given equality between groups i and i + 1. 
                dim 3, 4 and 5 - Represent the constraint coefficients of P

        '''
        # Precompute matrix S which selects the correct rows from P^T and
        # columns from M to compute element A_{ij}. First two dims are for
        # the dimensions of A, last three dims are over the coefficients
        # of P^T
        S = np.zeros(
                (self.n_classes, self.n_classes,
                self.n_classes, self.n_classes, self.n_groups)
        )
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                # selecting the ith row of M as the
                # coefficients of the jth column of P^T
                # for computing A_ij
                if len(M.shape) == 4:
                    # Case where there is a different 2 dim
                    # matrix M for each i and group as in predictive rate
                    # parity where the numerator is equal to \sum_k P_{ik} Z_{kj} e_i
                    # which makes the matrix M_{ikj} be Z_{kj} e_i with four
                    # dimensions total (one for group, i, k, and j)
                    S[i, j, :, j] = M[i, i, :]
                else:
                    # Case where there is only a single 2 dim
                    # Matrix M for all i
                    S[i, j, :, j] = M[i, :] 

        cons = np.zeros(
                (self.n_groups - 1, self.n_classes, self.n_classes,
                self.n_classes, self.n_classes, self.n_groups)
        )
        for g in range(self.n_groups - 1):
            cons[g, :, :, :, :, g] = S[:, :, :, :, g]
            cons[g, :, :, :, :, g + 1] = -S[:, :, :, :, g + 1]
        return cons    
    
    
    def adjust(self,
               goal='odds',
               loss='macro',
               round=4,
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
        self.constraints = [self.__get_constraints(self.p_vecs[i],
                                                   self.p_a[i],
                                                   self.cp_mats[i])
                            for i in range(self.n_groups)]
        
        # Arranging the constraint weights by group comparisons
        tpr_cons, fpr_cons, strict_cons, norm_cons = self.__pair_constraints(
            self.constraints
            )
        
        # First option is macro loss, or the sum of the unweighted
        # conditional probabilities from above
        if loss == 'macro':
            off_loss = [[np.delete(a, i, 0).sum(0) 
                         for i in range(self.n_classes)]
                        for a in self.cp_mats]
            self.obj = np.array(off_loss).flatten()
            
            '''
            self.obj = -1 * self.cp_mats.flatten()
            '''
        
        # Next option is weighted macro, which weights the conditional
        # sums by the baseline class (not group) probabilities
        elif loss == 'w_macro':
            pass
        
        # And finally is micro, which weights each 
        elif loss == 'micro':
            u = np.array([[np.delete(a, i, 0)
                           for i in range(self.n_classes)]
                          for a in self.cp_mats])
            p = np.array([[np.delete(a, i).reshape(-1, 1)
                           for i in range(self.n_classes)]
                          for a in self.p_vecs])
            w = np.array([[p[i, j] * u[i, j]
                           for j in range(self.n_classes)]
                          for i in range(self.n_groups)])
            self.w = w
            self.obj = w.sum(2).flatten()
        
        # Normazliation bounds; used in all scenarios
        norm_bounds = np.repeat(1, norm_cons.shape[0])
        
        # Choosing the constraint conditions
        if 'odds' in goal:
            con = np.concatenate([tpr_cons, fpr_cons, norm_cons])
            eo_bounds = np.repeat(0, tpr_cons.shape[0] * 2)
            con_bounds = np.concatenate([eo_bounds, norm_bounds])
        
        elif 'opportunity' in goal:
            con = np.concatenate([tpr_cons, norm_cons])
            tpr_bounds = np.repeat(0, tpr_cons.shape[0])
            con_bounds = np.concatenate([tpr_bounds, norm_bounds])
        
        elif 'strict' in goal:
            con = np.concatenate([strict_cons, norm_cons])
            strict_bounds = np.repeat(0, strict_cons.shape[0])
            con_bounds = np.concatenate([strict_bounds, norm_bounds])
        
        self.goal = goal
        self.con = con
        self.con_bounds = con_bounds
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=self.obj,
                                       bounds=[0, 1],
                                       A_eq=con,
                                       b_eq=con_bounds,
                                       method='highs')
        
        if self.opt.status == 0:
            # Getting the Y~ matrices
            self.m = tools.pars_to_cpmat(self.opt,
                                         n_groups=self.n_groups,
                                         n_classes=self.n_classes)
            
            # Calculating group-specific ROC scores from the new parameters
            self.rocs = tools.parmat_to_roc(self.m,
                                            self.p_vecs,
                                            self.cp_mats)
            self.loss = 1 - np.sum(self.p_y * self.rocs[0, :, 1])
            self.macro_loss = 1 - np.mean(self.rocs[0, :, 1])
        else:
            print('\nBalancing failed: Linear program is infeasible.\n')
            self.rocs = np.nan
            self.loss = np.nan
            self.macro_loss = np.nan
            
        if summary:
            self.summary(org=False)
    
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
             lp_lines='all', 
             shade_hull=True,
             chance_line=True,
             palette='colorblind',
             tight=False,
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
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Organizing the ROC points by group (rbg) and by outcome (rbo)
        rbg = [r for r in self.old_rocs.values()]
        rbo = [np.concatenate([r.loc[i].values.reshape(1, -1) 
                               for r in rbg], 0)
               for i in range(self.n_classes)]
        
        # Making a tall df so we can use sns.relplot()
        tall = deepcopy(rbg)
        for i, df in enumerate(tall):
            df['group'] = self.groups[i]
            df['outcome'] = self.outcomes
        tall = pd.concat(tall, axis=0)
        
        # Setting up the plots
        rp = sns.relplot(x='fpr', 
                         y='tpr', 
                         hue='group', 
                         col='outcome', 
                         data=tall,
                         kind='scatter',
                         palette=palette)
        rp.fig.set_tight_layout(tight)
        rp.set(xlim=xlim, ylim=ylim)
        
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
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting coordinates for the upper portions of the hulls
            for i, ax in enumerate(rp.axes[0]):
                g_r = pd.DataFrame(rbo[i], columns=['fpr', 'tpr'])
                group_var = np.array([[g]*3 for g in self.groups]).flatten()
                upper_x = np.array([[0, fpr, 1] 
                                    for fpr in g_r.fpr.values]).flatten()
                upper_y = np.array([[0, tpr, 1] 
                                    for tpr in g_r.tpr.values]).flatten()
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
                             ax=ax,
                             alpha=0.75, 
                             legend=False)
                
                # Optionally adding lower lines to complete the hulls
                if lp_lines == 'all':
                    lower_x = np.array([[0, 1 - fpr, 1] 
                                        for fpr in g_r.fpr.values]).flatten()
                    lower_y = np.array([[0, 1 - tpr, 1] 
                                        for tpr in g_r.tpr.values]).flatten()
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
                                 ax=ax,
                                 alpha=0.75, 
                                 legend=False)
                
                if shade_hull:
                    for i, group in enumerate(self.groups):
                        uc = upper_df[upper_df.group == group]
                        u_null = np.array([0, uc.x.values[1], 1])
                                            
                        if lp_lines == 'upper':
                            ax.fill_between(x=uc.x,
                                            y1=uc.y,
                                            y2=u_null,
                                            color=cmap[i],
                                            alpha=0.1) 
                        if lp_lines == 'all':
                            lc = lower_df[lower_df.group == group]
                            l_null = np.array([0, lc.x.values[1], 1])
                            ax.fill_between(x=uc.x,
                                            y1=uc.y,
                                            y2=u_null,
                                            color=cmap[i],
                                            alpha=0.1) 
                            ax.fill_between(x=lc.x,
                                             y1=l_null,
                                             y2=lc.y,
                                             color=cmap[i],
                                             alpha=0.1)
                
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.rocs is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            for i, ax in enumerate(rp.axes[0]):
                if ('odds' in self.goal) | ('strict' in self.goal):
                    ax.scatter(self.rocs[0, i, 0],
                               self.rocs[0, i, 1],
                               marker='x',
                               color='black')
                
                elif 'opportunity' in self.goal:
                    ax.hlines(self.rocs[0, i, 1],
                              xmin=0,
                              xmax=1,
                              color='black',
                              linestyles='--',
                              linewidths=0.5)
                
                elif 'parity' in self.goal:
                    pass
        
        # Optionally adding the chance line
        if chance_line:
            [ax.plot((0, 1), (0, 1), color='lightgray') 
             for ax in rp.axes[0]]
        
        plt.show()
    
    def summary(self, org=True, adj=True, round=4):
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        if org:
            org_loss = 1 - np.dot(self.p_y, np.diag(self.cp_mat))
            print('\nPre-adjustment group rates are \n')
            for i in range(self.n_groups):
                group_stats = tools.cpmat_to_roc(self.p_vecs[i],
                                                 self.cp_mats[i]).round(round)
                group_stats.index = self.outcomes
                print(self.groups[i])
                print(group_stats.to_string() + '\n')
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            # Casting the post-adjustment ROC scores as a DF
            if 'opportunity' in self.goal:
                print('\nPost-adjustment group rates are \n')
                for i, r in enumerate(self.rocs):
                    print(self.groups[i])
                    adj_coords = pd.DataFrame(r, columns=['fpr', 'tpr'])
                    adj_coords = adj_coords.round(round)
                    adj_coords.index = self.outcomes
                    print(adj_coords.to_string())
                    print('\n')
            else:
                adj_coords = pd.DataFrame(self.rocs[0],
                                          columns=['fpr', 'tpr']).round(round)
                adj_coords.index = self.outcomes
                print('\nPost-adjustment rates for all groups are \n')
                print(adj_coords.to_string())
            
            adj_loss = 1 - np.sum(self.p_y * adj_coords.tpr.values)
            print('\nAnd loss is %.4f\n' %adj_loss)

