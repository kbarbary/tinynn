"""Example datasets"""
import os
import gzip
import sys
from urllib.request import urlopen

import numpy as np

__all__ = ["load_mnist"]


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


MNIST_FILENAMES = {"train": ("train-images-idx3-ubyte",
                             "train-labels-idx1-ubyte"),
                   "test": ("t10k-images-idx3-ubyte",
                            "t10k-labels-idx1-ubyte")}

def fetch_mnist(subset, dirname):
    os.makedirs(dirname, exist_ok=True)
    root_url = "http://yann.lecun.com/exdb/mnist/"
    for fname in MNIST_FILENAMES[subset]:
        local = os.path.join(dirname, fname)
        if not os.path.exists(local):
            download_gzip_file(root_url + fname + '.gz', local)


def load_mnist(subset, dirname='data'):
    fetch_mnist(subset, dirname)
    return tuple(read_idx(os.path.join(dirname, fname))
                 for fname in MNIST_FILENAMES[subset])
