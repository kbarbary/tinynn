"""A minimal implementation of a dense neural net with an arbitrary
number of layers, backpropagation, and a few different activation functions."""

import numpy as np

# Activation and cost functions (with gradients)
def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y, y * (1.0 - y)


def relu(x):
    y = np.maximum(0.0, x)
    grad = (x > 0.0).astype(np.float64)
    return y, grad


def tanh(x):
    y = np.tanh(x)
    return y, 1.0 - y**2


def cross_entropy_cost(A, Y):
    m = Y.shape[1]
    cost = (1.0 / m) * np.sum(-Y * np.log(A) - (1.0 - Y) * np.log(1.0 - A))
    dA = (1.0 / m) * (-(Y / A) + (1.0 - Y) / (1.0 - A))  # dcost/dA
    return cost, dA


class Layer(object):
    def __init__(self, n_in: int, n_out: int, activation):
        self.g = activation
        self.W = np.random.normal(scale=0.01, size=(n_out, n_in))
        self.b = np.zeros((n_out, 1))
        self._cache = {}

    def __call__(self, X):
        """Forward propagation (and cache intermediate results)"""
        Z = self.W @ X + self.b
        A, dAdZ = self.g(Z)
        self._cache['X'] = X
        self._cache['Z'] = Z
        self._cache['dAdZ'] = dAdZ
        return A

    def backward(self, dA, alpha):
        """Backward propagation and update parameters"""
        dZ = dA * self._cache['dAdZ']
        dW = dZ @ self._cache['X'].T
        db = np.sum(dZ, axis=1, keepdims=True)
        dX = self.W.T @ dZ

        # update
        self.W -= alpha * dW
        self.b -= alpha * db

        return dX


class NeuralNetwork(object):
    def __init__(self, layer_sizes, activations):
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
                       for i in range(len(activations))]
        self.costs = []

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def train(self, X, Y, niter=100, alpha=0.05):
        for i in range(niter):
            A = self(X)
            cost, dA = cross_entropy_cost(A, Y)

            # backprop and update parameters via gradient descent
            for layer in reversed(self.layers):
                dA = layer.backward(dA, alpha)

            self.costs.append(cost)
            if not (i % 10):
                print(i, "cost =", cost)

        return self
