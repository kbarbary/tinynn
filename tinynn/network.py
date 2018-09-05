import time

import numpy as np
import tqdm

from .layer import Layer
from .optimizers import Descent
from .costs import CrossEntropy
from .preprocessing import partition

__all__ = ["Network"]

def _identity(x):
    return x

class Network(object):
    def __init__(self, *layers, x_preprocess=None, y_preprocess=None,
                 postprocess=None):
        self.layers = layers
        self.costs = []
        self.x_preprocess = x_preprocess or _identity
        self.y_preprocess = y_preprocess or _identity
        self.postprocess = postprocess or _identity

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def train(self, X, Y, niter=100, opt_type=Descent, opt_params={'α': 0.001},
              λ=0.0):

        # if X and Y are not batches, make them into batches.
        if not isinstance(X, list):
            X, Y = [X], [Y]

        # Run preprocessing on each batch
        X = [self.x_preprocess(batch) for batch in X]
        Y = [self.y_preprocess(batch) for batch in Y]

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

def predict(self, X):
    X = self.x_preprocess(X)
    Y = self(X)
    return self.postprocess(Y)

"""
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
"""
