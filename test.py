import numpy as np

import tinynn

# helpers
def ngradient(f, v):
    """
    Calculate the gradient of f with respect to array v.
    """
    g = np.zeros_like(v)
    δ = np.sqrt(np.finfo(v.dtype).eps)
    for i in range(v.size):
        tmp = v.flat[i]
        v.flat[i] = tmp - δ/2
        y1 = f()
        v.flat[i] = tmp + δ/2
        y2 = f()
        g.flat[i] += (y2 - y1) / δ
        v.flat[i] = tmp
    return g


class SumSine(tinynn.Layer):
    """Dummy layer used in checking gradients."""
    def __call__(self, X):
        self.X = X
        return np.sum(np.sin(X))

    def back(self, dA):
        return dA * np.cos(self.X)


def gradcheck(layer, X, verbose=False):
    """
    Check that the layer's numerical gradients match exact gradients for
    input X and the layer parameters.
    """
    agg = SumSine()

    # get exact gradients
    exact = {}
    agg(layer(X))  # forward
    exact['dX'] = layer.back(agg.back(1.0))
    for name in layer.param_names:
        exact['d'+name] = getattr(layer, 'd'+name).copy()

    # numerical gradients
    approx = {}
    approx['dX'] = ngradient(lambda: agg(layer(X)), X)
    for name in layer.param_names:
        approx['d'+name] = ngradient(lambda: agg(layer(X)),
                                     getattr(layer, name))

    for key in exact:
        ok = np.allclose(exact[key], approx[key], rtol=1e-5, atol=1e-5)
        if not ok or verbose:
            print(layer, key)
            print("exact:")
            print(exact[key])
            print("approx:")
            print(approx[key])
        if not ok:
            raise AssertionError("exact and numeric gradients do not match")


def test_densebatchnorm():
    X = np.random.rand(10, 10)
    layer = tinynn.DenseBatchNorm(10, 5, active=True)
    gradcheck(layer, X)

    layer = tinynn.DenseBatchNorm(10, 5, active=False)
    gradcheck(layer, X)


def test_conv():
    X = np.random.rand(3, 10, 10, 3)
    layer = tinynn.Conv(3, 5, (3, 3), stride=(1, 1))
    gradcheck(layer, X)

def test_pool():
    X = np.random.rand(2, 5, 5, 2)
    layer = tinynn.Pool((2, 2), stride=(1, 1))
    gradcheck(layer, X, verbose=True)
