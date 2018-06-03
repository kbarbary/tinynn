#!/usr/bin/env python
import os
import gzip
import sys
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt

import tinynn
from tinynn import Dense, DenseBatchNorm, ReLU, Sigmoid, Softmax


def download_gzip_file(url, file_name):
    response = gzip.GzipFile(fileobj=urlopen(url))
    with open(file_name, 'wb') as f:
        f.write(response.read())


def read_idx(fname):
    """Read an IDX format file into a numpy array. IDX is a very simple
    binary format described here: http://yann.lecun.com/exdb/mnist/"""
    with open(fname, 'rb') as f:
        # read magic bytes: dtype and ndim
        magic = f.read(4)
        assert magic[0:2] == b'\x00\x00'
        dtypes = {8: np.uint8, 9: np.int8, 11: np.int16,
                  12: np.int32, 13: np.float32, 14: np.float64}
        dtype = np.dtype(dtypes[magic[2]]).newbyteorder('>')
        ndim = magic[3]

        # read dimensions
        dims = []
        for i in range(ndim):
            b = f.read(4)
            dims.append(int.from_bytes(b, byteorder='big'))

        # read data
        data = np.fromfile(f, dtype=dtype, count=np.product(dims))
        data.shape = dims

    return data


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # get some data
    root_url = "http://yann.lecun.com/exdb/mnist/"
    names = {"train_images": "train-images-idx3-ubyte",
             "train_labels": "train-labels-idx1-ubyte",
             "test_images": "t10k-images-idx3-ubyte",
             "test_labels": "t10k-labels-idx1-ubyte"}
    fnames = {key: "data/" + name for key, name in names.items()}
    for key in names:
        if not os.path.exists(fnames[key]):
            download_gzip_file(root_url + names[key] + '.gz', fnames[key])

    # Read the data
    data = {key: read_idx(fname) for key, fname in fnames.items()}

    # Flatten features
    X = {}
    for k in ('train', 'test'):
        images = data[k + '_images']
        X[k] = images.reshape((images.shape[0], -1)).T

    # normalize inputs
    normalize = tinynn.normalizer(X['train'])
    X['train'] = normalize(X['train'])
    X['test'] = normalize(X['test'])

    # one-hot encode Y labels
    Y = {}
    for k in ('train', 'test'):
        labels = data[k + '_labels']
        y = np.zeros((labels.size, labels.max() + 1))
        y[np.arange(labels.size), labels] = 1.0
        Y[k] = y.T

    if len(sys.argv) > 1 and sys.argv[1] == 'gradcheck':
        print("checking gradient...")
        X = np.random.rand(10, 10)
        Y = np.random.rand(3, 10)
        network = tinynn.Network(DenseBatchNorm(10, 5, active=True), ReLU(),
                                 DenseBatchNorm(5, 3, active=True), Softmax())
        network.gradcheck(X, Y)
        exit()


    Xs, Ys = tinynn.partition(X['train'], Y['train'], size=1000)

    # Run training
    network = tinynn.Network(DenseBatchNorm(28*28, 100), ReLU(),
                             DenseBatchNorm(100, 10), Softmax())
    network.train(Xs, Ys, niter=100,
                  opt_type=tinynn.ADAM, opt_params={'α': 0.0005})

    # show training cost
    plt.plot(network.costs)
    plt.ylim(ymin=0.0)
    plt.ylabel("cost")
    plt.xlabel("iteration")
    plt.savefig("costs.png")

    # Validate
    for l in network.layers:
        l.active = False

    for key in ('train', 'test'):
        print(key, 'set')
        Ypred = network(X[key])
        labels = np.argmax(Ypred, axis=0)

        print("Truth:     ", data[key + '_labels'][:30])
        print("Prediction:", labels[:30])

        correct = data[key + '_labels'] == labels
        print("Correct: {:6.2f}%\n".format(100.0 * correct.mean()))
