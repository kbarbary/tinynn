"""
A minimal implementation of a dense neural net with an arbitrary
number of layers, backpropagation, and a few different activation functions.
"""

import time

import numpy as np
import tqdm


# -----------------------------------------------------------------------------
# helpers and preprocessing

def sum1(X):
    return np.sum(X, axis=1, keepdims=True)


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


# -----------------------------------------------------------------------------
# Activation and cost functions (with gradients)

class Layer(object):
    def __init__(self):
        self.param_names = []

class Sigmoid(Layer):
    def __call__(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y
    def back(self, dY):
        dydx = self.y * (1.0 - self.y)
        return dY * dydx


class ReLU(Layer):
    def __call__(self, x):
        self.x = x
        return np.maximum(0.0, x)
    def back(self, dY):
        dydx = 1.0 * (self.x > 0.0)
        return dY * dydx


class Tanh(Layer):
    def __call__(self, x):
        self.y = np.tanh(x)
        return self.y
    def back(self, dY):
        dydx = 1.0 - self.y**2
        return dY * dydx


# cross entropy cost for two-class classification where predictions
# are scalars in [0, 1]
def binomial_cross_entropy_cost(A, Y):
    m = Y.shape[1]
    cost = (1.0 / m) * np.sum(-Y * np.log(A) - (1.0 - Y) * np.log(1.0 - A))
    dA = (1.0 / m) * (-(Y / A) + (1.0 - Y) / (1.0 - A))  # dcost/dA
    return cost, dA


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


# optimizers / update policies

class Descent(object):
    def __init__(self, layer, param_name, α=0.001):
        self.α = α
        self.layer = layer
        self.param_name = param_name

    def update(self):
        v = getattr(self.layer, self.param_name)
        grad = getattr(self.layer, 'd'+self.param_name)
        v -= self.α * grad


class ADAM(object):
    def __init__(self, layer, param_name, α=0.001, β1=0.9, β2=0.999, ϵ=1e-8):
        self.α, self.β1, self.β2, self.ϵ = α, β1, β2, ϵ
        self.layer = layer
        self.param_name = param_name
        v = getattr(self.layer, self.param_name)
        self.vgrad = np.zeros_like(v)
        self.sgrad = np.zeros_like(v)
        self.β1t = 1.0  # tracks β1^t (initially t=0)
        self.β2t = 1.0  # tracks β2^t (initially t=0)

    def update(self):
        v = getattr(self.layer, self.param_name)
        grad = getattr(self.layer, 'd'+self.param_name)
        self.vgrad = self.β1 * self.vgrad + (1 - self.β1) * grad
        self.sgrad = self.β2 * self.sgrad + (1 - self.β2) * grad**2
        self.β1t *= self.β1
        self.β2t *= self.β2
        corr1 = 1 - self.β1t
        corr2 = 1 - self.β2t
        v -= self.α * ((self.vgrad / corr1) /
                       (np.sqrt(self.sgrad / corr2) + self.ϵ))


class Network(object):
    def __init__(self, *layers):
        self.layers = layers
        self.costs = []

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def train(self, X, Y, niter=100, opt_type=Descent, opt_params={'α': 0.001},
              λ=0.0):

        # if X and Y are not batches, make them into batches.
        if not isinstance(X, list):
            X, Y = [X], [Y]

        # Define cost function
        costfn = CrossEntropy()

        # get an optimizer for each parameter
        optimizers = []
        for layer in self.layers:
            for param_name in layer.param_names:
                optimizers.append(opt_type(layer, param_name, **opt_params))

        pbar = tqdm.trange(niter)
        try:
            for i in pbar:
                for Xi, Yi in zip(X, Y):
                    m = Xi.shape[1]  # number of training examples

                    # forward propagation
                    cost = costfn(self(Xi), Yi)
                    if λ != 0.0:
                        for layer in self.layers:
                            cost += λ / (2 * m) * np.sum(layer.W ** 2)

                    # backward propagation
                    dA = costfn.back()
                    for layer in reversed(self.layers):
                        dA = layer.back(dA)
                        if λ != 0.0:
                            layer.dW += (λ / m) * layer.W

                    # update params
                    for optimizer in optimizers:
                        optimizer.update()

                self.costs.append(cost)
                pbar.set_postfix(cost=cost)

        except KeyboardInterrupt:
            pass
        pbar.close()

        return self


    def gradcheck(self, X, Y):
        costfn = CrossEntropy()

        # run cost function once with backprop to fill layer gradients
        cost = costfn(self(X), Y)
        dA = costfn.back()
        for layer in reversed(self.layers):
            dA = layer.back(dA)

        for layer in self.layers:
            for name in layer.param_names:
                v = getattr(layer, name)
                exactg = getattr(layer, 'd'+name)
                g = np.zeros_like(v)
                δ = np.sqrt(np.finfo(v.dtype).eps)
                for i in range(v.size):
                    tmp = v.flat[i]
                    v.flat[i] = tmp - δ/2
                    y1 = costfn(self(X), Y)
                    v.flat[i] = tmp + δ/2
                    y2 = costfn(self(X), Y)
                    g.flat[i] = (y2 - y1) / δ
                    v.flat[i] = tmp

                ok = np.allclose(g, exactg, rtol=1e-5, atol=1e-5)
                if not ok:
                    print(layer, name)
                    print("exact:")
                    print(exactg)
                    print("numeric:")
                    print(g)
                    raise Exception()
