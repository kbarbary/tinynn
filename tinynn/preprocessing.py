import numpy as np

__all__ = ["partition", "normalizer", "onehot"]


def partition(X, Y, axis=-1, size=256):
    """
    For ndarrays shuffle along axis and split into chunks of of size ``size``
    along same axis.
    """
    m = X.shape[axis]
    assert  Y.shape[axis] == m
    xslices = X.ndim * [slice(None, None)]
    yslices = Y.ndim * [slice(None, None)]

    # purmute both arrays
    idx = np.random.permutation(m)
    xslices[axis] = idx
    yslices[axis] = idx
    Xp = X[xslices]
    Yp = Y[yslices]

    Xs = []
    Ys = []
    start = 0
    while start < m:
        end = min(start+size, m)
        xslices[axis] = slice(start, end)
        yslices[axis] = slice(start, end)
        Xs.append(Xp[xslices])
        Ys.append(Yp[yslices])
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
