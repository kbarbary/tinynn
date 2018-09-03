"""Convolution and Pooling layers"""
import numpy as np

from .layer import Layer

__all__ = ["Conv", "Pool"]


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
                for x in range(nx):
                    for c in range(nc):
                        # slice in input volume
                        y_in_min = y * self.stride[0]
                        y_in_max = y_in_min + fy
                        x_in_min = x * self.stride[1]
                        x_in_max = x_in_min + fx

                        # 3-d slice of input volume to be convolved
                        x_slice = X_pad[i, y_in_min:y_in_max, x_in_min:x_in_max, :]

                        # convolve
                        Z[i, y, x, c] = np.sum(x_slice * self.W[:, :, :, c]) + self.b[0, 0, 0, c]

        # save input for backprop
        self.X = X

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
                        xslice = xslice(x0, x0 + fx)

                        dX_pad[i, yslice, xslice, :] += W[:, :, :, c] * dZ[i, y, x, c]
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
                        A[i, y, x, c] = self.agg(x_slice)

        return A
