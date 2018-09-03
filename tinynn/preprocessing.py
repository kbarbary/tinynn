import numpy as np

__all__ = ["partition", "normalizer", "onehot"]


def partition(X, Y, size=256):
    """
    For 2-d arrays of trailing dimension size ``m``, shuffle along
    trailing dimension and split dimension into chunks of size ``size``.
    """
    m = X.shape[1]
    assert  Y.shape[1] == m
    idx = np.random.permutation(m)
    Xp = X[:, idx]
    Yp = Y[:, idx]
    Xs = []
    Ys = []
    start = 0
    while start < m:
        end = min(start+size, m)
        Xs.append(Xp[:, start:end])
        Ys.append(Yp[:, start:end])
        start += size
    return Xs, Ys


def normalizer(X):
    """
    Return a function that would normalize the array X along axis=1.

    Parameters
    ----------
    X : ndarray
        An array of shape `(n_features, n_examples)`. The returned function
        makes the mean of each feature (column) zero and the standard
        deviation of each feature (column) one.
    """
    mu = np.mean(X, axis=1, keepdims=True)
    sig = np.sqrt(np.var(X, axis=1, keepdims=True) + 1e-8)
    return lambda X: (X - mu) / sig


def onehot(x, dtype=float):
    """
    One-hot encode a 1-d array of non-negative integers.
    """
    out = np.zeros((x.size, x.max() + 1), dtype=dtype)
    out[np.arange(x.size), x] = 1
    return out.T
