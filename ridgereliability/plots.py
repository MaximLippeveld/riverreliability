# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/plots.ipynb (unless otherwise specified).

__all__ = ['ridge_reliability_diagram', 'class_wise_ridge_reliability_diagram', 'confidence_reliability_diagram',
           'class_wise_confidence_reliability_diagram']

# Cell

from ridgereliability import utils, metrics as rmetrics

import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib import gridspec, cm

import numpy as np
from scipy.stats import beta
from mclearn.performance import get_beta_parameters, beta_avg_pdf

import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

# Internal Cell

def _decorate_ax(ax:matplotlib.axes.Axes):
    """Apply cosmetic changes to a matplotlib axis.

    Arguments:
    ax -- matplotlib axis
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.spines.values(), color=cm.tab20c(18))
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=cm.tab20c(18))

# Internal Cell

def ridge_diagram(beta_distributions_per_bin:np.array, proportions_per_bin:np.array, plot_densities:bool, ax:matplotlib.axes.Axes):

    _decorate_ax(ax)

    class clipped_cm:
        def __init__(self, n, base_cm=cm.Greys):
            self.n = n
            self.space = np.linspace(0.2, 0.8, n+1)
            self.cm = [base_cm(p) for p in self.space]

        def __call__(self, x):
            return self.cm[int(x*self.n)]
    cmap = clipped_cm(len(proportions_per_bin))

    y_min = 0-1/(len(beta_distributions_per_bin)/2)
    y_max = 1+1/(len(beta_distributions_per_bin)/2)
    ax.set_ylim(y_min, y_max)

    sorted_idx = np.argsort(proportions_per_bin)

    ax.plot([0,1], [0,1], color=cm.tab20c(19), zorder=0)

    for i, (proportion, dist) in enumerate(zip(proportions_per_bin[sorted_idx], beta_distributions_per_bin[sorted_idx])):
        n_layers = 4
        layer = [len(proportions_per_bin)*n_layers+1 - (n_layers*i)+n for n in range(n_layers)]

        if proportion is np.nan:
            continue

        if len(dist) == 2:
            # dist contains the parameters of the beta distribution
            a, b = dist

            # sample the beta
            x = np.linspace(0, 1, 500)
            beta_norm = beta.pdf(x, a, b)
        else:
            # dist contains samples of the beta pdf

            ## sample from the beta distribution using the pdf probabilities

            # make it impossible to sample 0 or 1
            # in theory this should never happen, but approximations introduce errors
            dist[0] = 0.0
            dist[-1] = 0.0

            x = np.linspace(0, 1, len(dist))
            samples = np.random.choice(x, size=500, p=dist/dist.sum())

            ## fit a beta distribution to the samples
            a, b, loc, scale = beta.fit(samples, floc=0, fscale=1)

            beta_norm = dist

        prob_interval = beta.interval(0.95, a, b)
        dist_mean = a/(a+b)

        # rescale it to 0-x range
        beta_norm /= beta_norm.max()
        beta_norm /= len(proportions_per_bin)/2

        if not plot_densities:
            # plot probability interval line
            ax.plot([prob_interval[0], prob_interval[1]], [proportion, proportion], lw=1, color=cmap(1-proportion), zorder=layer[1])
        else:
            # plot densities if wanted
            ax.plot(x, beta_norm+proportion, lw=1, linestyle="dotted", color=cmap(1-proportion), zorder=layer[1])
            # ax.plot([0, 1], [proportion, proportion], color=cmap(1-proportion), linestyle="dotted", lw=1, alpha=0.5, zorder=layer[1])

            idx = [j for j,p in enumerate(x) if prob_interval[0] <= p <= prob_interval[1]]
            ax.plot(x[idx], beta_norm[idx]+proportion, 'r-', lw=1.5, color=cmap(1-proportion), zorder=layer[3])
            ax.plot(x[idx], beta_norm[idx]+proportion, 'r-', lw=4, color="white", zorder=layer[2])

        # plot extra marker at distribution mode
        ax.scatter(dist_mean, proportion, color=cmap(1-proportion), edgecolor="white", linewidth=2, s=25, zorder=layer[2])

# Internal Cell
def _add_metrics_to_title(ax:matplotlib.axes.Axes, metrics:list, y_probs:np.array, y_preds:np.array, y_true:np.array):
    title = ax.get_title()
    if len(title) > 0:
        title += " - "

    for metric in metrics:
        title += f"{metric.__name__}: {metric(y_probs, y_preds, y_true):.3f}, "

    ax.set_title(title[:-2])

# Cell

def ridge_reliability_diagram(y_probs:np.array, y_preds:np.array, y_true:np.array, ax:matplotlib.axes.Axes, bins="fd", plot_densities:bool=True, exact:bool=False):

    ax.set_ylabel("Confidence level")
    ax.set_xlabel("Posterior balanced accuracy")

    num_classes = len(np.unique(y_true))

    if len(y_probs.shape) == 2:
        if y_probs.shape[1] == 2:
            y_probs = y_probs[:, 0]
        else:
            y_probs = y_probs.max(axis=1)

    bin_indices, edges = utils.get_bin_indices(y_probs, bins, 0.0, 1.0, return_edges=True)
    unique_bin_indices = sorted(np.unique(bin_indices))

    proportions = np.empty((len(unique_bin_indices),), dtype=np.float32) # store mean confidence

    if not exact:
        n_samples = 10000
        distributions = np.empty((len(unique_bin_indices), n_samples), dtype=np.float32) # store beta parameters
        x = np.linspace(0, 1, n_samples)
    else:
        distributions = np.empty((len(unique_bin_indices), 2), dtype=np.int)

    # compute beta distributions per bin per class
    for i, bin_idx in enumerate(unique_bin_indices):
        selector = bin_indices == bin_idx

        proportions[i] = y_probs[selector].mean()

        if not exact:
            conf = confusion_matrix(y_true[selector], y_preds[selector], labels=np.arange(0, num_classes))
            parameters = get_beta_parameters(conf)
            distributions[i] = beta_avg_pdf(x, parameters)
        else:
            correct = (y_true[selector] == y_preds[selector]).sum()
            incorrect = len(y_true[selector]) - correct
            distributions[i] = correct + 1, incorrect + 1

    ridge_diagram(distributions, proportions, plot_densities, ax)

# Cell

def class_wise_ridge_reliability_diagram(y_probs, y_preds, y_true, axes:matplotlib.axes.Axes=None, bins="fd", plot_densities=True, metric=rmetrics.peace, show_k_least_calibrated=None):

    classes = np.unique(y_true)

    if show_k_least_calibrated is None:
        show_k_least_calibrated = len(classes)

    plots = min(show_k_least_calibrated, len(classes))

    if axes is None:
        fig, axes = plt.subplots(1, plots, subplot_kw={"aspect": 0.75}, constrained_layout=True, sharex=True, sharey=True, dpi=72)
    assert len(axes) == plots, f"Wrong amount of axes provided: {plots} needed, but {len(axes)} provided."

    y_true_binarized = label_binarize(y_true, classes=classes)
    y_preds_binarized = label_binarize(y_preds, classes=classes)

    metric_values = []
    for c in classes:
        probs = np.where(y_preds_binarized[:, c]==0, 1-y_probs[:, c], y_probs[:, c])
        metric_values.append(metric(probs, y_preds_binarized[:, c], y_true_binarized[:, c]))

    for ax, c in zip(axes, np.argsort(metric_values)[::-1][:show_k_least_calibrated]):
        probs = np.where(y_preds_binarized[:, c]==0, 1-y_probs[:, c], y_probs[:, c])

        ax.set_title(f"Class {c}")

        ridge_reliability_diagram(probs, y_preds_binarized[:, c], y_true_binarized[:, c], ax, bins, plot_densities, exact=True)

# Internal Cell

def bar_diagram(edges:np.array, bin_accuracies:np.array, bin_confidences:np.array, ax:matplotlib.axes.Axes):
    """Plot a bar plot confidence reliability diagram.

    Arguments:
    edges -- Edges of the probability bins
    bin_accuracies -- Accuracy per bin
    bin_confidences -- Average confidence of predictions in bin
    ax -- Axes on which the diagram will be plotted (will be decorated by `_decorate_ax`)
    """

    _decorate_ax(ax)

    ax.plot([0,1], [0,1], linestyle="--", color=cm.tab20c(16))

    for xi, yi, bi in zip(edges, bin_accuracies, bin_confidences):
        if bi is np.nan:
            continue
        if yi < 0:
            continue

        ax.bar(xi, yi, width=edges[1], align="edge", color=cm.tab20c(18), edgecolor=cm.tab20c(19), linewidth=2, zorder=0)
        if yi >= bi:
            bar = ax.bar(xi+edges[1]/2, np.abs(bi-yi), bottom=bi, width=edges[1]/4, align="center", color=cm.tab20c(17), zorder=1)
        else:
            bar = ax.bar(xi+edges[1]/2, np.abs(bi-yi), bottom=yi, width=edges[1]/4, align="center", color=cm.tab20c(17), zorder=1)

# Cell

def confidence_reliability_diagram(y_probs:np.array, y_preds:np.array, y_true:np.array, ax:matplotlib.axes.Axes, bins="fd", balanced:bool=True):
    """Plot a confidence reliability diagram.

    Arguments:
    y_probs -- Array containing prediction confidences
    y_preds -- Array containing predicted labels (shape (N,))
    y_true -- Array containing true labels (shape (N,))
    ax -- Axes on which the diagram will be plotted (will be decorated by `_decorate_ax`)
    bins -- Description of amount of bins in which to divide prediction confidences (see `numpy.histogram_bin_edges` for options)
    balanced -- Flag for using balanced accuracy score
    """

    if len(y_probs.shape) == 2:
        if y_probs.shape[1] == 2:
            y_probs = y_probs[:, 0]
        else:
            y_probs = y_probs.max(axis=1)

    bin_indices, edges = utils.get_bin_indices(y_probs, bins, 0.0, 1.0, return_edges=True)
    unique_bin_indices = sorted(np.unique(bin_indices))

    mean_confidences = np.full((len(edges)-1,), dtype=np.float32, fill_value=np.nan)
    bin_metric = np.full((len(edges)-1,), dtype=np.float32, fill_value=np.nan)

    metric = balanced_accuracy_score if balanced else accuracy_score

    ax.set_xlabel("Confidence level")
    ax.set_ylabel("Balanced accuracy" if balanced else "Accuracy")

    for bin_idx in unique_bin_indices:
        selector = bin_indices == bin_idx

        C = confusion_matrix(y_true[selector], y_preds[selector])
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(C) / C.sum(axis=1)
        per_class = per_class[~np.isnan(per_class)]
        if len(per_class) < 2:
            continue

        mean_confidences[bin_idx-1] = np.mean(y_probs[selector])
        bin_metric[bin_idx-1] = metric(y_true[selector], y_preds[selector])

        bar_diagram(edges, bin_metric, mean_confidences, ax)

# Cell

def class_wise_confidence_reliability_diagram(y_probs:np.array, y_preds:np.array, y_true:np.array, axes:matplotlib.axes.Axes, bins="fd", balanced:bool=True):
    """Plot a class-wise confidence reliability diagram.

    Arguments:
    y_probs -- Array containing prediction confidences
    y_preds -- Array containing predicted labels (shape (N,))
    y_true -- Array containing true labels (shape (N,))
    ax -- Axes on which the diagram will be plotted (will be decorated by `_decorate_ax`)
    bins -- Description of amount of bins in which to divide prediction confidences (see `numpy.histogram_bin_edges` for options)
    balanced -- Flag for using balanced accuracy score
    """

    classes = np.unique(y_true)

    y_true_binarized = label_binarize(y_true, classes=classes)
    y_preds_binarized = label_binarize(y_preds, classes=classes)

    for ax, c in zip(axes, range(len(classes))):
        ax.set_title(f"Class {c}")
        probs = np.where(y_preds_binarized[:, c]==0, 1-y_probs[:, c], y_probs[:, c])
        confidence_reliability_diagram(probs, y_preds_binarized[:, c], y_true_binarized[:, c], ax, bins, balanced)