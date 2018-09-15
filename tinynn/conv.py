"""Convolution and Pooling layers"""
import numpy as np

from .layer import Layer

__all__ = ["Conv", "Pool", "Flatten", "Transpose"]


class Conv(Layer):
    def __init__(self, n_in, n_out, kernel_shape, stride=(1, 1), pad='same'):
        if pad == 'same':
            # (left, right) padding in each dimenison on input
            self.pad = ((0, 0),
                        ((kernel_shape[0]-1)//2, kernel_shape[0]//2),
                        ((kernel_shape[1]-1)//2, kernel_shape[1]//2),
                        (0, 0))
        else:
            raise ValueError("only pad='same' supported.")
        self.stride = stride
        weight_shape = kernel_shape + (n_in, n_out)
        self.W = np.random.normal(scale=np.sqrt(2 / n_in), size=weight_shape)
        self.b = np.zeros((1, 1, 1, n_out))
        self.param_names = ['W', 'b']

    def __call__(self, X):
        """
        Forward pass.

        Parameters
        ----------
        X : shape (m, n_y, n_x, n_c)
            Where m is number of examples, n_y is height, n_x is width,
            n_c is input channels.
        """
        # pad input
        X_pad = np.pad(X, self.pad, 'constant')

        # get some dimensions
        m, ny_in, nx_in, nc_in = X.shape
        fy, fx, nc_in, nc = self.W.shape
        ypad_tot = sum(self.pad[1])
        xpad_tot = sum(self.pad[2])

        # compute dimensions of output volume
        ny = (ny_in - fy + ypad_tot) // self.stride[0] + 1
        nx = (nx_in - fx + xpad_tot) // self.stride[1] + 1

        # create output volume
        Z = np.zeros((m, ny, nx, nc))

        # loop over output volume
        for i in range(m):  # loop over batch of training examples
            for y in range(ny):
                # y slice in input volume
                y_in_min = y * self.stride[0]
                yslice = slice(y_in_min, y_in_min + fy)
                for x in range(nx):
                    # x slice in input volume
                    x_in_min = x * self.stride[1]
                    xslice = slice(x_in_min, x_in_min + fx)
                    
                    # 3-d slice of input volume to be convolved
                    # but add a length-1 dimension to the end
                    # making it 4-d
                    X_slice = X_pad[i, yslice, xslice, :, None]
                    Z[i, y, x, :] = np.sum(X_slice * self.W, axis=(0, 1, 2)) + self.b[0, 0, 0, :]

        # save input for backprop
        self.X = X
        return Z

    def back(self, dZ):
        m, ny, nx, nc = dZ.shape   # layer output dimensions
        fy, fx, nc_in, nc = self.W.shape
        X = self.X  # layer input

        # initialize outputs
        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # pad X and dX
        X_pad = np.pad(X, self.pad, 'constant')
        dX_pad = np.pad(dX, self.pad, 'constant')

        # loop over output volume
        for i in range(m):
            for y in range(ny):
                for x in range(nx):
                    for c in range(nc):
                        # slice in input volume
                        y0 = y * self.stride[0]
                        yslice = slice(y0, y0 + fy)
                        x0 = x * self.stride[1]
                        xslice = slice(x0, x0 + fx)

                        dX_pad[i, yslice, xslice, :] += self.W[:, :, :, c] * dZ[i, y, x, c]
                        dW[:, :, : ,c] += X_pad[i, yslice, xslice, :] * dZ[i, y, x, c]
                        db[:, :, :, c] += dZ[i, y, x, c]

        # set dX to the right slice of dX_pad
        ypad, xpad = self.pad[1], self.pad[2]
        dX = dX_pad[:, ypad[0]:-ypad[1], xpad[0]:-xpad[1], :]

        self.dW = dW
        self.db = db
        return dX


class Pool(Layer):
    def __init__(self, kernel_shape, stride=(1, 1), mode='max'):
        self.stride = stride
        self.kernel_shape = kernel_shape
        if mode not in ('max', 'mean'):
            raise ValueError("mode must be 'max' or 'mean'")
        self.agg = getattr(np, mode)
        self.param_names = []

    def __call__(self, X):
        m, ny_in, nx_in, nc_in = X.shape
        fy, fx = self.kernel_shape

        # dimensions of output
        ny = int(1 + (ny_in - fy) / self.stride[0])
        nx = int(1 + (nx_in - fx) / self.stride[1])
        nc = nc_in

        # initialize output
        A = np.zeros((m, ny, nx, nc))

        for i in range(m):
            for y in range(ny):
                for x in range(nx):
                    for c in range(nc):
                        # slice in input volume
                        y_in_min = y * self.stride[0]
                        yslice = slice(y_in_min, y_in_min + fy)
                        x_in_min = x * self.stride[1]
                        xslice = slice(x_in_min, x_in_min + fx)

                        X_slice = X[i, yslice, xslice, c]
                        A[i, y, x, c] = self.agg(X_slice)

        self.X = X
        self.A = A
        return A

    def back(self, dA):
        m, ny_in, nx_in, nc_in = self.X.shape
        fy, fx = self.kernel_shape

        # dimensions of output
        ny = int(1 + (ny_in - fy) / self.stride[0])
        nx = int(1 + (nx_in - fx) / self.stride[1])
        nc = nc_in

        # initialize gradient
        dX = np.zeros_like(self.X)

        # loop over *output*
        for i in range(m):
            for y in range(ny):
                for x in range(nx):
                    for c in range(nc):
                        # slice in input volume
                        y_in_min = y * self.stride[0]
                        yslice = slice(y_in_min, y_in_min + fy)
                        x_in_min = x * self.stride[1]
                        xslice = slice(x_in_min, x_in_min + fx)

                        a = self.A[i, y, x, c]  # output value
                        da = dA[i, y, x, c]
                        if self.agg is np.max:
                            mask = self.X[i, yslice, xslice, c] == a
                            dX[i, yslice, xslice, c] += da / mask.sum() * mask
                        elif self.agg is np.mean:
                            dX[i, yslice, xslice, c] += da / (fx * fy)
        return dX


class Flatten(Layer):
    """For a input of shape (m, n1, n2, ...) transform to
    (m, n1*n2*...)."""
    def __init__(self, keepdim='first'):
        if keepdim not in ('first', 'last'):
            raise ValueError("keepdim must be 'first' or 'last'.")
        self.keepdim = keepdim
        self.param_names = []

    def __call__(self, X):
        if self.keepdim == 'first':
            return X.reshape((X.shape[0], -1))
        elif self.keepdim == 'last':
            return X.reshape((-1, X.shape[-1]))
        self.in_shape = X.shape

    def back(self, dA):
        dA.reshape(self.in_shape)


class Transpose(Layer):
    def __call__(self, X):
        return X.T
    def back(self, dA):
        return dA.T
