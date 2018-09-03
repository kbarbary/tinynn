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
        g.flat[i] = (y2 - y1) / δ
        v.flat[i] = tmp
    return g


class SumSine(tinynn.Layer):
    """Dummy layer used in checking gradients."""
    def __call__(self, X):
        self.X = X
        return np.sum(np.sin(X))

    def back(self, dA):
        return dA * np.cos(self.X)


def gradcheck(layer, X):
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
        if not ok:
            print(layer, name)
            print("exact:")
            print(exactg)
            print("numeric:")
            print(g)
            raise AssertionError("exact and numeric gradients to not match")


def test_densebatchnorm():
    X = np.random.rand(10, 10)
    layer = tinynn.DenseBatchNorm(10, 5, active=True)
    gradcheck(layer, X)

    layer = tinynn.DenseBatchNorm(10, 5, active=False)
    gradcheck(layer, X)
