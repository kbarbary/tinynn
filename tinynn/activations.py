import numpy as np

from .layer import Layer

__all__ = ["Sigmoid", "ReLU", "Tanh"]


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
