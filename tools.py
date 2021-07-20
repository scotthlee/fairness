import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import binom, chi2, norm, percentileofscore
from itertools import combinations
from copy import deepcopy
from multiprocessing import Pool
from copy import deepcopy


class CLFRates:
    def __init__(self,
                 y, 
                 y_,
                 round=4):
        self.tab = confusion_matrix(y, y_)
        tn = self.tab[0, 0]
        fn = self.tab[1, 0]
        fp = self.tab[0, 1]
        tp = self.tab[1, 1]
        self.pr = np.round((tp + fp) / len(y), round)
        self.nr = np.round((tn + fn) / len(y), round)
        self.tnr = np.round(tn / (tn + fp), round)
        self.tpr = np.round(tp / (tp + fn), round)
        self.fnr = np.round(fn / (fn + tp), round)
        self.fpr = np.round(fp / (fp + tn), round)
        self.acc = (tn + tp) / len(y)


def make_multi_predictor(y, p, catvar=None):
    out = deepcopy(y)
    
    if catvar is not None:
        if p.shape[0] != len(np.unique(catvar)):
            print('Please provide conditional probs for all groups.')
            return
        if len(y) != len(catvar):
            print('Labels should be the same length as groups.')
            return
        else:
            cats = np.unique(catvar)
            c_ids = [np.where(catvar == c)[0] for c in cats]
            for i, ids in enumerate(c_ids):
                preds = make_multi_predictor(y[ids], p[i])
                out[ids] = preds
            return out
            
    y_c = np.unique(y)
    y_dict = dict(zip(list(range(len(y_c))), y_c))
    y_c_ids = [np.where(y == c)[0] for c in y_c]
    
    for i, ids in enumerate(y_c_ids):
        if np.sum(p[i]) != 1:
            p[i] /= np.sum(p[i])
        n = len(ids)
        one_hot = np.random.multinomial(1, p[i], n)
        cols = np.argmax(one_hot, axis=1)
        cats = [y_dict[c] for c in cols]
        out[ids] = cats
    
    return out
    
    

def make_predictor(y, tpr, fpr, catvar=None):
    out = np.zeros(shape=len(y))
    
    if catvar is not None:
        if len(tpr) != len(np.unique(catvar)):
            print('Please provide TPR and FPR for each group.')
            return
        if len(y) != len(catvar):
            print('Labels should be the same length as groups.')
            return
        else:
            cats = np.unique(catvar)
            c_ids = [np.where(catvar == c)[0] for c in cats]
            for i, ids in enumerate(c_ids):
                preds = make_predictor(y[ids], tpr[i], fpr[i])
                out[ids] = preds
            return out.astype(np.uint8)
            
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    out[pos] = np.random.binomial(1, tpr, len(pos))
    out[neg] = np.random.binomial(1, fpr, len(neg))
    return out.astype(np.uint8)


def make_catvar(n, p, levels):
    cat_dict = dict(zip(list(range(len(p))), levels))
    one_hot = np.random.multinomial(1, p, n)
    out = np.argmax(one_hot, axis=1)
    out = np.array([cat_dict[c] for c in out])
    return out


def make_label(p, catvar, levels=None):
    out = np.zeros(len(catvar))
    cats = np.unique(catvar)
    c_ids = [np.where(catvar == c)[0] for c in cats]
    for i, ids in enumerate(c_ids):
        n = len(ids)
        probs = p[i]
        if len(probs) > 1:
            if np.sum(probs) != 1:
                probs /= np.sum(probs)
            labs = np.random.multinomial(1, probs, n)
            preds = np.argmax(labs, axis=1)
        else:
            preds = np.random.binomial(1, p[i], n)
        out[ids] = preds
    if levels is not None:
        l_dict = dict(zip(np.unique(out), levels))
        out = np.array([l_dict[c] for c in out])
        return out
    else:
        return out.astype(np.uint8)


def loss_from_roc(y, probs, roc):
    points = [(roc[0][i], roc[1][i]) for i in range(len(roc[0]))]
    guess_list = [threshold(probs, t) for t in roc[2]]
    accs = [accuracy_score(y, g) for g in guess_list]
    js = [p[1] - p[0] for p in points]
    tops = [from_top(point) for point in points]
    return {'guesses': guess_list, 
            'accs': accs, 
            'js': js, 
            'tops': tops}


def from_top(roc_point, round=4):
    d = np.sqrt(roc_point[0]**2 + (roc_point[1] - 1)**2)
    return d


def roc_coords(y, y_, round=4):
    # Getting hte counts
    tab = confusion_matrix(y, y_)
    tn = tab[0, 0]
    fn = tab[1, 0]
    fp = tab[0, 1]
    tp = tab[1, 1]
    
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


# Quick function for thresholding probabilities
def threshold(probs, cutoff=.5):
    return np.array(probs >= cutoff).astype(np.uint8)


# Calculates McNemar's chi-squared statistic
def mcnemar_test(true, pred, cc=True):
    cm = confusion_matrix(true, pred)
    b = int(cm[0, 1])
    c = int(cm[1, 0])
    if cc:
        stat = (abs(b - c) - 1)**2 / (b + c)
    else:
        stat = (b - c)**2 / (b + c)
    p = 1 - chi2(df=1).cdf(stat)
    outmat = np.array([b, c, stat, p]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['b', 'c', 'stat', 'pval'])
    return out


# Calculates the Brier score for multiclass problems
def brier_score(true, pred):
    n_classes = len(np.unique(true))
    assert n_classes > 1
    if n_classes == 2:
        bs = np.sum((pred - true)**2) / true.shape[0]
    else:
        y = onehot_matrix(true)
        row_diffs = np.diff((pred, y), axis=0)[0]
        squared_diffs = row_diffs ** 2
        row_sums = np.sum(squared_diffs, axis=1) 
        bs = row_sums.mean()
    return bs


# Runs basic diagnostic stats on categorical predictions
def clf_metrics(true, 
                pred,
                average='weighted',
                preds_are_probs=False,
                cutpoint=0.5,
                mod_name=None,
                round=4,
                round_pval=False,
                mcnemar=False,
                argmax_axis=1):
    '''Runs basic diagnostic stats on binary (only) predictions'''
    # Converting pd.Series to np.array
    stype = type(pd.Series())
    if type(pred) == stype:
        pred = pred.values
    if type(true) == stype:
        true = true.values
    
    # Optional exit for doing averages with multiclass/label inputs
    if len(np.unique(true)) > 2:
        # Getting binary metrics for each set of results
        codes = np.unique(true)
        
        # Argmaxing for when we have probabilities
        if preds_are_probs:
            auc = roc_auc_score(true,
                                pred,
                                average=average,
                                multi_class='ovr')
            brier = brier_score(true, pred)
            pred = np.argmax(pred, axis=argmax_axis)
        
        # Making lists of the binary predictions (OVR)    
        y = [np.array([doc == code for doc in true], dtype=np.uint8)
             for code in codes]
        y_ = [np.array([doc == code for doc in pred], dtype=np.uint8)
              for code in codes]
        
        # Getting the stats for each set of binary predictions
        stats = [clf_metrics(y[i], y_[i], round=16) for i in range(len(y))]
        stats = pd.concat(stats, axis=0)
        stats.fillna(0, inplace=True)
        cols = stats.columns.values
        
        # Calculating the averaged metrics
        if average == 'weighted':
            weighted = np.average(stats, 
                                  weights=stats.true_prev,
                                  axis=0)
            out = pd.DataFrame(weighted).transpose()
            out.columns = cols
        elif average == 'macro':
            out = pd.DataFrame(stats.mean()).transpose()
        elif average == 'micro':
            out = clf_metrics(np.concatenate(y),
                              np.concatenate(y_))
        
        # Adding AUC and AP for when we have probabilities
        if preds_are_probs:
            out.auc = auc
            out.brier = brier
        
        # Rounding things off
        out = out.round(round)
        count_cols = [
                      'tp', 'fp', 'tn', 'fn', 'true_prev',
                      'pred_prev', 'prev_diff'
        ]
        out[count_cols] = out[count_cols].round()
        
        if mod_name is not None:
            out['model'] = mod_name
        
        return out
    
    # Thresholding the probabilities, if provided
    if preds_are_probs:
        auc = roc_auc_score(true, pred)
        brier = brier_score(true, pred)
        ap = average_precision_score(true, pred)
        pred = threshold(pred, cutpoint)
    else:
        brier = np.round(brier_score(true, pred), round)
    
    # Constructing the 2x2 table
    confmat = confusion_matrix(true, pred)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]
    
    # Calculating the main binary metrics
    ppv = np.round(tp / (tp + fp), round) if tp + fp > 0 else 0
    sens = np.round(tp / (tp + fn), round) if tp + fn > 0 else 0
    spec = np.round(tn / (tn + fp), round) if tn + fp > 0 else 0
    npv = np.round(tn / (tn + fn), round) if tn + fn > 0 else 0
    f1 = np.round(2*(sens*ppv) / (sens+ppv), round) if sens + ppv != 0 else 0 
    
    # Calculating the Matthews correlation coefficient
    mcc_num = ((tp * tn) - (fp * fn))
    mcc_denom = np.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = mcc_num / mcc_denom if mcc_denom != 0 else 0
    
    # Calculating Youden's J and the Brier score
    j = sens + spec - 1
    
    # Rolling everything so far into a dataframe
    outmat = np.array([tp, fp, tn, fn,
                       sens, spec, ppv,
                       npv, j, f1, mcc, brier]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['tp', 'fp', 'tn', 
                                'fn', 'sens', 'spec', 'ppv',
                                'npv', 'j', 'f1', 'mcc', 'brier'])
    
    # Optionally tacking on stats from the raw probabilities
    if preds_are_probs:
        out['auc'] = auc
        out['ap'] = ap
    else:
        out['auc'] = 0
        out['ap'] = 0
    
    # Calculating some additional measures based on positive calls
    true_prev = int(np.sum(true == 1))
    pred_prev = int(np.sum(pred == 1))
    abs_diff = (true_prev - pred_prev) * -1
    rel_diff = np.round(abs_diff / true_prev, round)
    if mcnemar:
        pval = mcnemar_test(true, pred).pval[0]
        if round_pval:
            pval = np.round(pval, round)
    count_outmat = np.array([true_prev, pred_prev, abs_diff, 
                             rel_diff]).reshape(-1, 1)
    count_out = pd.DataFrame(count_outmat.transpose(),
                             columns=['true_prev', 'pred_prev', 
                                      'prev_diff', 'rel_prev_diff'])
    out = pd.concat([out, count_out], axis=1)
    
    # Optionally dropping the mcnemar p-val
    if mcnemar:
        out['mcnemar'] = pval
    
    # And finally tacking on the model name
    if mod_name is not None:
        out['model'] = mod_name
    
    return out


def jackknife_metrics(targets, 
                      guesses, 
                      average='weighted'):
    # Replicates of the dataset with one row missing from each
    rows = np.array(list(range(targets.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]

    # using a pool to get the metrics across each
    scores = [clf_metrics(targets[idx],
                          guesses[idx],
                          average=average) for idx in j_rows]
    scores = pd.concat(scores, axis=0)
    means = scores.mean()
    
    return scores, means


def boot_stat_cis(stat,
                  jacks,
                  boots,
                  a=0.05,
                  exp=False,
                  method="bca",
                  interpolation="nearest",
                  transpose=True,
                  outcome_axis=1,
                  stat_axis=2):
    # Renaming because I'm lazy
    j = jacks
    n = len(boots)
    
    # Calculating the confidence intervals
    lower = (a / 2) * 100
    upper = 100 - lower

    # Making sure a valid method was chosen
    methods = ["pct", "diff", "bca"]
    assert method in methods, "Method must be pct, diff, or bca."

    # Calculating the CIs with method #1: the percentiles of the
    # bootstrapped statistics
    if method == "pct":
        cis = np.nanpercentile(boots,
                               q=(lower, upper),
                               interpolation=interpolation,
                               axis=0)
        cis = pd.DataFrame(cis.transpose(),
                           columns=["lower", "upper"],
                           index=colnames)

    # Or with method #2: the percentiles of the difference between the
    # obesrved statistics and the bootstrapped statistics
    elif method == "diff":
        diffs = stat - boots
        percents = np.nanpercentile(diffs,
                                    q=(lower, upper),
                                    interpolation=interpolation,
                                    axis=0)
        lower_bound = pd.Series(stat_vals + percents[0])
        upper_bound = pd.Series(stat_vals + percents[1])
        cis = pd.concat([lower_bound, upper_bound], axis=1)
        cis = cis.set_index(stat.index)

    # Or with method #3: the bias-corrected and accelerated bootstrap
    elif method == "bca":
        # Calculating the bias-correction factor
        n_less = np.sum(boots < stat, axis=0)
        p_less = n_less / n
        z0 = norm.ppf(p_less)

        # Fixing infs in z0
        z0[np.where(np.isinf(z0))[0]] = 0.0

        # Estiamating the acceleration factor
        diffs = j[1] - j[0]
        numer = np.sum(np.power(diffs, 3))
        denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3/2)

        # Getting rid of 0s in the denominator
        zeros = np.where(denom == 0)[0]
        for z in zeros:
            denom[z] += 1e-6

        # Finishing up the acceleration parameter
        acc = numer / denom

        # Calculating the bounds for the confidence intervals
        zl = norm.ppf(a / 2)
        zu = norm.ppf(1 - (a / 2))
        lterm = (z0 + zl) / (1 - acc * (z0 + zl))
        uterm = (z0 + zu) / (1 - acc * (z0 + zu))
        ql = norm.cdf(z0 + lterm) * 100
        qu = norm.cdf(z0 + uterm) * 100

        # Returning the CIs based on the adjusted quantiles;
        # I know this code is hideous
        if len(boots.shape) > 2:
            n_outcomes = range(boots.shape[outcome_axis])
            n_vars = range(boots.shape[stat_axis])
            cis = np.array([
                [np.nanpercentile(boots[:, i, j],
                                  q =(ql[i][j], 
                                      qu[i][j]),
                                  axis=0) 
                                  for i in n_outcomes]
                for j in n_vars
            ])
        else:
            n_stats = range(len(ql))
            cis = np.array([
                np.nanpercentile(boots[:, i],
                                 q=(ql[i], qu[i]),
                                 interpolation=interpolation,
                                 axis=0) 
                for i in n_stats])
        
        # Optional exponentiation for log-link models
        if exp:
            cis = np.exp(cis)
        
        # Optional transposition
        if transpose:
            cis = cis.transpose()

    return cis


# Calculates bootstrap confidence intervals for an estimator
class boot_cis:
    def __init__(
        self,
        targets,
        guesses,
        n=100,
        a=0.05,
        group=None,
        method="bca",
        interpolation="nearest",
        average='weighted',
        mcnemar=False,
        seed=10221983):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series())
        if type(targets) == stype:
            targets = targets.values
        if type(guesses) == stype:
            guesses = guesses.values

        # Getting the point estimates
        stat = clf_metrics(targets,
                           guesses,
                           average=average,
                           mcnemar=mcnemar).transpose()

        # Pulling out the column names to pass to the bootstrap dataframes
        colnames = list(stat.index.values)

        # Making an empty holder for the output
        scores = pd.DataFrame(np.zeros(shape=(n, stat.shape[0])),
                              columns=colnames)

        # Setting the seed
        if seed is None:
            seed = np.random.randint(0, 1e6, 1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, n)

        # Generating the bootstrap samples and metrics
        boots = [boot_sample(targets, seed=seed) for seed in seeds]
        scores = [clf_metrics(targets[b], 
                              guesses[b], 
                              average=average) for b in boots]
        scores = pd.concat(scores, axis=0)

        # Calculating the confidence intervals
        lower = (a / 2) * 100
        upper = 100 - lower

        # Making sure a valid method was chosen
        methods = ["pct", "diff", "bca"]
        assert method in methods, "Method must be pct, diff, or bca."

        # Calculating the CIs with method #1: the percentiles of the
        # bootstrapped statistics
        if method == "pct":
            cis = np.nanpercentile(scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=["lower", "upper"],
                               index=colnames)

        # Or with method #2: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == "diff":
            stat_vals = stat.transpose().values.ravel()
            diffs = stat_vals - scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(stat_vals + percents[0])
            upper_bound = pd.Series(stat_vals + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
            cis = cis.set_index(stat.index)

        # Or with method #3: the bias-corrected and accelerated bootstrap
        elif method == "bca":
            # Calculating the bias-correction factor
            stat_vals = stat.transpose().values.ravel()
            n_less = np.sum(scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)

            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0

            # Estiamating the acceleration factor
            j = jackknife_metrics(targets, guesses)
            diffs = j[1] - j[0]
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3 / 2)

            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6

            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j

            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            lterm = (z0 + zl) / (1 - acc * (z0 + zl))
            uterm = (z0 + zu) / (1 - acc * (z0 + zu))
            ql = norm.cdf(z0 + lterm) * 100
            qu = norm.cdf(z0 + uterm) * 100
            
            # Passing things back to the class
            self.acc = acc.values
            self.b = z0
            self.ql = ql
            self.qu = qu

            # Returning the CIs based on the adjusted quintiles
            cis = [
                np.nanpercentile(
                    scores.iloc[:, i],
                    q=(ql[i], qu[i]),
                    interpolation=interpolation,
                    axis=0,
                ) for i in range(len(ql))
            ]
            cis = pd.DataFrame(cis, 
                               columns=["lower", "upper"], 
                               index=colnames)

        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ["stat", "lower", "upper"]

        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores

        return


def average_pvals(p_vals, 
                  w=None, 
                  method='harmonic',
                  smooth=True,
                  smooth_val=1e-7):
    if smooth:
        p = p_vals + smooth_val
    else:
        p = deepcopy(p_vals)
    if method == 'harmonic':
        if w is None:
            w = np.repeat(1 / len(p), len(p))
        p_avg = 1 / np.sum(w / p)
    elif method == 'fisher':
        stat = -2 * np.sum(np.log(p))
        p_avg = 1 - chi2(df=1).cdf(stat)
    return p_avg


def jackknife_sample(X):
    rows = np.array(list(range(X.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]
    return j_rows


# Generates bootstrap indices of a dataset with the option
# to stratify by one of the (binary-valued) variables
def boot_sample(df,
                by=None,
                size=None,
                seed=None,
                return_df=False):
    
    # Setting the random states for the samples
    if seed is None:
        seed = np.random.randint(1, 1e6, 1)[0]
    np.random.seed(seed)
    
    # Getting the sample size
    if size is None:
        size = df.shape[0]
    
    # Sampling across groups, if group is unspecified
    if by is None:
        np.random.seed(seed)
        idx = range(size)
        boot = np.random.choice(idx,
                                size=size,
                                replace=True)
    
    # Sampling by group, if group has been specified
    else:
        levels = np.unique(by)
        n_levels = len(levels)
        level_idx = [np.where(by == level)[0]
                     for level in levels]
        boot = np.random.choice(range(n_levels),
                                size=n_levels,
                                replace=True)
        boot = np.concatenate([level_idx[i] for i in boot]).ravel()
    
    if not return_df:
        return boot
    else:
        return df.iloc[boot, :]
    

class diff_boot_cis:
    def __init__(self,
                 ref,
                 comp,
                 a=0.05,
                 abs_diff=False,
                 method='bca',
                 interpolation='nearest'):
        # Quick check for a valid estimation method
        methods = ['pct', 'diff', 'bca']
        assert method in methods, 'Method must be pct, diff, or bca.'
        
        # Pulling out the original estiamtes
        ref_stat = pd.Series(ref.cis.stat.drop('true_prev').values)
        ref_scores = ref.scores.drop('true_prev', axis=1)
        comp_stat = pd.Series(comp.cis.stat.drop('true_prev').values)
        comp_scores = comp.scores.drop('true_prev', axis=1)
        
        # Optionally Reversing the order of comparison
        diff_scores = comp_scores - ref_scores
        diff_stat = comp_stat - ref_stat
            
        # Setting the quantiles to retrieve
        lower = (a / 2) * 100
        upper = 100 - lower
        
        # Calculating the percentiles 
        if method == 'pct':
            cis = np.nanpercentile(diff_scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose())
        
        elif method == 'diff':
            diffs = diff_stat.values.reshape(1, -1) - diff_scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(diff_stat + percents[0])
            upper_bound = pd.Series(diff_stat + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
        
        elif method == 'bca':
            # Removing true prevalence from consideration to avoid NaNs
            ref_j_means = ref.jack[1].drop('true_prev')
            ref_j_scores = ref.jack[0].drop('true_prev', axis=1)
            comp_j_means = comp.jack[1].drop('true_prev')
            comp_j_scores = comp.jack[0].drop('true_prev', axis=1)
            
            # Calculating the bias-correction factor
            n = ref.scores.shape[0]
            stat_vals = diff_stat.transpose().values.ravel()
            n_less = np.sum(diff_scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)
            
            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0
            
            # Estiamating the acceleration factor
            j_means = comp_j_means - ref_j_means
            j_scores = comp_j_scores - ref_j_scores
            diffs = j_means - j_scores
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3/2)
            
            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6
            
            acc = numer / denom
            
            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a/2))
            lterm = (z0 + zl) / (1 - acc*(z0 + zl))
            uterm = (z0 + zu) / (1 - acc*(z0 + zu))
            ql = norm.cdf(z0 + lterm) * 100
            qu = norm.cdf(z0 + uterm) * 100
                                    
            # Returning the CIs based on the adjusted quantiles
            cis = [np.nanpercentile(diff_scores.iloc[:, i], 
                                    q=(ql[i], qu[i]),
                                    interpolation=interpolation,
                                    axis=0) 
                   for i in range(len(ql))]
            cis = pd.DataFrame(cis, columns=['lower', 'upper'])
                    
        cis = pd.concat([ref_stat, comp_stat, diff_stat, cis], 
                        axis=1)
        cis = cis.set_index(ref_scores.columns.values)
        cis.columns = ['ref', 'comp', 'd', 
                       'lower', 'upper']
        
        # Passing stuff back up to return
        self.cis = cis
        self.scores = diff_scores
        self.b = z0
        self.acc = acc
        
        return


def grid_metrics(targets,
                 guesses,
                 step=.01,
                 min=0.0,
                 max=1.0,
                 by='f1',
                 average='binary',
                 counts=True):
    cutoffs = np.arange(min, max, step)
    if len((guesses.shape)) == 2:
        if guesses.shape[1] == 1:
            guesses = guesses.flatten()
        else:
            guesses = guesses[:, 1]
    if average == 'binary':
        scores = []
        for i, cutoff in enumerate(cutoffs):
            threshed = threshold(guesses, cutoff)
            stats = clf_metrics(targets, threshed)
            stats['cutoff'] = pd.Series(cutoff)
            scores.append(stats)
    
    return pd.concat(scores, axis=0)


def roc_cis(rocs, alpha=0.05, round=2):
    # Getting the quantiles to make CIs
    lq = (alpha / 2) * 100
    uq = (1 - (alpha / 2)) * 100
    fprs = np.concatenate([roc[0] for roc in rocs], axis=0)
    tprs = np.concatenate([roc[1] for roc in rocs], axis=0)
    roc_arr = np.concatenate([fprs.reshape(-1, 1), 
                              tprs.reshape(-1, 1)], 
                             axis=1)
    roc_df = pd.DataFrame(roc_arr, columns=['fpr', 'tpr'])
    roc_df.fpr = roc_df.fpr.round(round)
    unique_fprs = roc_df.fpr.unique()
    fpr_idx = [np.where(roc_df.fpr == fpr)[0] for fpr in unique_fprs]
    tpr_quants = [np.percentile(roc_df.tpr[idx], q=(lq, 50, uq)) 
                  for idx in fpr_idx]
    tpr_quants = np.vstack(tpr_quants)
    quant_arr = np.concatenate([unique_fprs.reshape(-1, 1),
                                tpr_quants],
                               axis=1)
    quant_df = pd.DataFrame(quant_arr, columns=['fpr', 'lower',
                                                'med', 'upper'])
    quant_df = quant_df.sort_values('fpr')
    return quant_df


# Converts a boot_cis['cis'] object to a single row
def merge_cis(c, round=4, mod_name=''):
    str_cis = c.round(round).astype(str)
    str_paste = pd.DataFrame(str_cis.stat + ' (' + str_cis.lower + 
                                 ', ' + str_cis.upper + ')',
                                 columns=[mod_name]).transpose()
    return str_paste


def merge_ci_list(l, mod_names=None, round=4):
    if type(l[0] != type(pd.DataFrame())):
        l = [c.cis for c in l]
    if mod_names is not None:
        merged_cis = [merge_cis(l[i], round, mod_names[i])
                      for i in range(len(l))]
    else:
        merged_cis = [merge_cis(c, round=round) for c in l]
    
    return pd.concat(merged_cis, axis=0)


def risk_ratio(y, pred, round=2):
    props = np.array(prop_table(y, pred, round=None))
    rr = props[1, 1] / props[1, 0]
    if round is not None:
        rr = np.round(rr, round)
    return rr


def odds_ratio(y, pred, round=2):
    tab = np.array(pd.crosstab(y, pred))
    OR = (tab[0, 0]*tab[1, 1]) / (tab[1, 0]*tab[0, 1])
    if round is not None:
        OR = np.round(OR, round)
    return OR


def cp_mat(y, y_):
    '''Returns the matrix of conditional probabilities y_ | y'''
    tab = pd.crosstab(y, y_).values
    probs = tab.transpose() / tab.sum(axis=1)
    return probs.transpose()


def p_vec(y, flatten=True):
    '''Returns the matrix of probabilities for the levels y'''
    tab = pd.crosstab(y, 'count').values
    out = tab / tab.sum()
    if flatten:
        out = out.flatten()
    return out


def constraint_weights(p_vec, cp_mat):
    '''Calculates TPR and FPR weights for the constraint matrix'''
    # Shortening the vars to keep things clean
    p = p_vec
    M = cp_mat
    
    # Setting up the matrix of parameters
    n_classes = M.shape[0]
    n_params = n_classes**2
    tpr = np.zeros(shape=(n_classes, n_params))
    fpr = np.zeros(shape=(n_classes, n_params))
    
    # Getting the combinations of outcomes
    combos = list(combinations(range(n_classes), n_classes - 1))
    combos = [list(c) for c in combos]
    
    # Getting weights for the first n-1 classes
    for i, c in enumerate(combos):
        start = i * (n_classes)
        end = start + n_classes
        fpr[i, start:end] = np.dot(p[c], M[c]) / p[c].sum()
        tpr[i, start:end] = M[i]
    '''
    # Getting the weights for the last class in terms of the others
    c = combos[0]
    last_tpr = M[n_classes - 1]
    last_fpr = np.dot(p[c], M[c]) / p[c].sum()
    
    # And filling in the last row of the constraint matrix
    fpr[n_classes - 1] = -1 * np.tile(last_fpr, n_classes - 1)
    tpr[n_classes - 1] = -1 * np.tile(last_tpr, n_classes - 1)
    '''
    return tpr, fpr


class ProbabilityMatrices():
    def __init__(self,
                 y,
                 y_,
                 a):
        '''Runs p_vec, cp_mat, and constraint_weights for each group'''
        # Getting some basic info for each group
        self.groups = np.unique(a)
        self.n_groups = len(self.groups)
        self.group_probs = p_vec(a)
        group_ids = [np.where(a == g) for g in self.groups]
        
        # Getting the group-specific P(Y), P(Y- | Y), and constraint matrices
        self.p_vecs = np.array([p_vec(y[ids]) for ids in group_ids])
        self.cp_mats = np.array([cp_mat(y[ids], y_[ids]) 
                                 for ids in group_ids])
        self.constraints = np.array([constraint_weights(self.p_vecs[i],
                                                        self.cp_mats[i]) 
                                     for i in range(self.n_groups)])
        
        # Weighting and then adding the class-specific TPRs for each group
        self.tpr_sums = [np.dot(self.p_vecs[i], self.constraints[i][0])
                         for i in range(self.n_groups)]
        
        # Weighting the group-specific TPRs by the group probabilities
        out = np.array([self.tpr_sums[i] * self.group_probs[i]
                        for i in range(self.n_groups)])
        
        # Flattening the weights so they work with the linear program
        self.loss_weights = out.flatten() * -1
        
        return


def constraint_pairs(pms):
    '''Makes a matrix of the pairwise constraints'''
    # Setting up the preliminaries
    tprs = pms.constraints[:, 0, :]
    fprs = pms.constraints[:, 1, :]
    n_params = tprs.shape[2]
    n_classes = tprs.shape[1]
    n_groups = pms.n_groups
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
    
    # Filling in the constraint comparisons
    for i, c in enumerate(group_combos):
        # Getting the original diffs
        diffs = pms.cp_mats[c[0]] - pms.cp_mats[c[1]]
        tpr_flip = np.sign(np.diag(diffs)).reshape(-1, 1)
        print(tpr_flip)
        print(tprs[c[0]])
        
        # Filling in the 
        tpr_cons[i, c[0]] = np.multiply(tpr_flip, tprs[c[0]])
        tpr_cons[i, c[1]] = np.multiply(tpr_flip, -1 * tprs[c[1]])
        fpr_cons[i, c[0]] = fprs[c[0]]
        fpr_cons[i, c[1]] = -1 * fprs[c[1]]
    
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
    
    return tpr_cons, fpr_cons, one_cons


def pars_to_cpmat(opt, n_group=3, n_class=3):
    '''Reshapes the LP parameters as an n_group * n_class * n_class array'''
    shaped = np.reshape(opt.x, (n_group, n_class, n_class))
    flipped = np.array([m.T for m in shaped])
    return flipped
