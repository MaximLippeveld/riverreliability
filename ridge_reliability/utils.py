# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/utils.ipynb (unless otherwise specified).

__all__ = ['freedman_diaconis', 'get_bin_indices', 'binning']

# Cell

import numpy as np
import scipy.stats

# Cell

def freedman_diaconis(values):
    """Compute the number of bins according to the Freedman-Diaconis rule.

    Parameters:
    values (np.array): values of the histogram

    Returns:
    bins (int): number of bins to use for the histogram of values
    """

    hist, bin_edges = np.histogram(values, bins='fd')
    return len(bin_edges) - 1


def get_bin_indices(y_probs, bins='fd', lower=None, upper=None, return_edges=False):
    """Compute a function across.

    Parameters:
    y_probs (np.array): predicted class probabilities
    bins (int or sequence of scalars or str, optional): number of bins
    return_edges (bool): return the edges used for the binning

    Returns:
    bin_indices (np.array): array that maps instances to bins
    edges (np.array): bin edges if return_edges is True

    """

    # check inputs
    assert len(y_probs.shape) == 1 and y_probs.dtype in [np.float, np.float32], 'Predicted class probabilties should be an array of floats'
    assert all(y_probs >= 0) and all(y_probs <= 1), 'Predicted class probabilities should lie between 0 and 1'

    # compute the bins
    if lower is None:
        lower = y_probs.min()
    if upper is None:
        upper = y_probs.max()

    edges = np.histogram_bin_edges(y_probs, bins=bins, range=(lower, upper))

    if not isinstance(bins, int):
        bins = len(edges) - 1

    # bin the confidence levels
    bin_indices = np.digitize(y_probs, edges, right=True)

    if return_edges:
        return bin_indices, edges

    return bin_indices


def binning(y_probs, y_preds, y_true, bin_indices, bin_func):
    """Compute a function across bins of confidence levels.

    Parameters:
    y_probs (np.array): predicted class probabilities
    y_preds (np.array): predicted class labels
    y_true (np.array): true class labels
    bin_indices (np.array): array that maps instances to bins (as obtained by `utils.get_bin_indices`)
    bin_func (lambda): function to compute for each bin

    Returns:
    result (float): result of the computation across bins

    """

    # check inputs
    assert len(y_probs.shape) == 1 and y_probs.dtype in [np.float, np.float32], 'Predicted class probabilties should be an array of floats'
    assert all(y_probs >= 0) and all(y_probs <= 1), 'Predicted class probabilities should lie between 0 and 1'
    assert len(y_preds.shape) == 1 and y_preds.dtype == np.int, 'Predicted class labels should be an array of integers'
    assert len(y_true.shape) == 1 and y_true.dtype == np.int, 'True class labels should be an array of integers'

    result = 0.
    for i in np.unique(bin_indices):
        y_probs_bin, y_preds_bin, y_true_bin = y_probs[bin_indices==i], y_preds[bin_indices==i], y_true[bin_indices==i]

        # update current estimate
        result += len(y_probs_bin) / y_probs.shape[0] * bin_func(y_probs_bin, y_preds_bin, y_true_bin)
    return result
