import numpy as np

from .layer import Layer


__all__ = ["Softmax", "CrossEntropy"]

class Softmax(Layer):
    """Numerically stable softmax along axis=0 of x"""
    def __call__(self, X):
        A = np.exp(X - np.max(X, axis=0, keepdims=True))
        A /= np.sum(A, axis=0, keepdims=True)
        self.A = A
        return A

    def back(self, dA):
        # Derivation: https://deepnotes.io/softmax-crossentropy
        return self.A * (dA - np.sum(self.A * dA, axis=0, keepdims=True))


class CrossEntropy(Layer):
    def __call__(self, A, Y):
        m = Y.shape[1]
        self.A, self.Y, self.m = A, Y, m
        return -(1.0 / m) * np.sum(Y * np.log(A))
    def back(self, dA=1.0):
        return -(1.0 / self.m) * self.Y / self.A * dA


# cross entropy cost for two-class classification where predictions
# are scalars in [0, 1]
def binomial_cross_entropy_cost(A, Y):
    m = Y.shape[1]
    cost = (1.0 / m) * np.sum(-Y * np.log(A) - (1.0 - Y) * np.log(1.0 - A))
    dA = (1.0 / m) * (-(Y / A) + (1.0 - Y) / (1.0 - A))  # dcost/dA
    return cost, dA
