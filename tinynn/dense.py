import numpy as np

from .layer import Layer

__all__ = ['Dense', 'BatchNorm']


def sum0(X):
    return np.sum(X, axis=0, keepdims=True)


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, bias=True):
        self.W = np.random.normal(scale=np.sqrt(2 / n_in), size=(n_in, n_out))
        self.param_names = ['W']
        self.bias = bias
        if self.bias:
            self.b = np.zeros((1, n_out))
            self.param_names.append('b')

    def __call__(self, X):
        self.X = X
        Z = X @ self.W
        if self.bias:
            Z += self.b
        return Z

    def back(self, dZ):
        self.dW = self.X.T @ dZ
        if self.bias:
            self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T


class BatchNorm(Layer):
    """Batch Normalization Layer.
    
    For each of `n` nodes, forces the mean of the examples to be zero and
    the variance of examples to be one. Examples are along the leading
    axis: Input and output shape are (m, n)
    """
    def __init__(self, n, ϵ=1e-8, momentum=0.9, active=True):
        self.γ = np.ones((1, n))
        self.β = np.zeros((1, n))
        self.μ = np.zeros((1, n))  # moving mean
        self.σ2 = np.ones((1, n))  # moving variance
        self.ϵ = ϵ
        self.momentum = momentum
        self.param_names = ['γ', 'β']
        self.active = active

    def __call__(self, Z):
        if self.active:
            μ = np.mean(Z, axis=0, keepdims=True)  # shape (1, n)
            resid = Z - μ
            σ2 = np.mean(resid**2, axis=0, keepdims=True)  # shape (1, n)
            Znorm = resid / np.sqrt(σ2 + self.ϵ) # shape (m, n)

            # update moving mean, variance
            self.μ = self.momentum * self.μ + (1 - self.momentum) * μ
            self.σ2 = self.momentum * self.σ2 + (1 - self.momentum) * σ2

            # cache for backprop
            self.μ_last = μ
            self.σ2_last = σ2
        else:
            Znorm = (Z - self.μ) / np.sqrt(self.σ2 + self.ϵ)

        # cache for backprop
        self.Z = Z
        self.Znorm = Znorm

        return self.γ * Znorm + self.β

    def back(self, dZp):
        Znorm = self.Znorm

        self.dβ = sum0(dZp)  # shape (1, n)
        self.dγ = sum0(dZp * Znorm) # shape (1, n)
        dZnorm = dZp * self.γ  # shape (m, n)

        if self.active:
            # this is nontrivial because μ and σ2 depend on Z
            μ, σ2, Z = self.μ_last, self.σ2_last, self.Z
            ϵ = self.ϵ
            m = Z.shape[0]
            denom = np.sqrt(σ2 + ϵ)
            dZ = ((m * dZnorm - sum0(dZnorm) - Znorm * sum0(dZnorm * Znorm)) /
                  (m * denom))

        else:
            dZ = dZnorm / np.sqrt(self.σ2 + self.ϵ)

        return dZ
