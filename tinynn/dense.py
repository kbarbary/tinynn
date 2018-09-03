import numpy as np

from .layer import Layer

__all__ = ['Dense', 'DenseBatchNorm']


def sum1(X):
    return np.sum(X, axis=1, keepdims=True)


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int):
        self.W = np.random.normal(scale=np.sqrt(2 / n_in), size=(n_out, n_in))
        self.b = np.zeros((n_out, 1))
        self.param_names = ['W', 'b']

    def __call__(self, X):
        self.X = X
        return self.W @ X + self.b

    def back(self, dZ):
        self.dW = dZ @ self.X.T
        self.db = np.sum(dZ, axis=1, keepdims=True)
        return self.W.T @ dZ


class DenseBatchNorm(Layer):
    def __init__(self, n_in, n_out, ϵ=1e-8, momentum=0.9, active=True):
        self.W = np.random.normal(scale=np.sqrt(2 / n_in), size=(n_out, n_in))
        self.γ = np.ones((n_out, 1))
        self.β = np.zeros((n_out, 1))
        self.μ = np.zeros((n_out, 1))  # moving mean
        self.σ2 = np.ones((n_out, 1))  # moving variance
        self.ϵ = ϵ
        self.momentum = momentum
        self.param_names = ['W', 'γ', 'β']
        self.active = active

    def __call__(self, X):
        Z = self.W @ X
        if self.active:
            μ = np.mean(Z, axis=1, keepdims=True)
            resid = Z - μ
            σ2 = np.mean(resid**2, axis=1, keepdims=True)
            Znorm = resid / np.sqrt(σ2 + self.ϵ)

            # update moving mean, variance
            self.μ = self.momentum * self.μ + (1 - self.momentum) * μ
            self.σ2 = self.momentum * self.σ2 + (1 - self.momentum) * σ2

            # cache for backprop
            self.μ_last = μ
            self.σ2_last = σ2
        else:
            Znorm = (Z - self.μ) / np.sqrt(self.σ2 + self.ϵ)

        # cache for backprop
        self.X = X
        self.Z = Z
        self.Znorm = Znorm

        return self.γ * Znorm + self.β

    def back(self, dZp):
        Znorm = self.Znorm

        self.dβ = sum1(dZp)
        self.dγ = sum1(dZp * Znorm)
        dZnorm = dZp * self.γ

        if self.active:
            # this is nontrivial because μ and σ2 depend on Z
            μ, σ2, Z = self.μ_last, self.σ2_last, self.Z
            ϵ = self.ϵ
            m = Z.shape[1]
            denom = np.sqrt(σ2 + ϵ)
            dZ = ((m * dZnorm - sum1(dZnorm) - Znorm * sum1(dZnorm * Znorm)) /
                  (m * denom))

        else:
            dZ = dZnorm / np.sqrt(self.σ2 + self.ϵ)

        self.dW = dZ @ self.X.T
        return self.W.T @ dZ
