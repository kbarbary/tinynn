"""optimizers / update policies"""
import numpy as np

__all__ = ["Descent", "ADAM"]


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
