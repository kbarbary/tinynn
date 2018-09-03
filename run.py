#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import tinynn
from tinynn import Dense, DenseBatchNorm, ReLU, Sigmoid, Softmax


# read data
X_train, y_train = tinynn.load_mnist('train')
X_test, y_test = tinynn.load_mnist('test')

# flatten data
X = {'train': X_train.reshape((X_train.shape[0], -1)).T,
     'test': X_test.reshape((X_test.shape[0], -1)).T}

# normalize inputs
normalize = tinynn.normalizer(X['train'])
X['train'] = normalize(X['train'])
X['test'] = normalize(X['test'])

# one-hot encode labels
Y = {'train': tinynn.onehot(y_train),
     'test': tinynn.onehot(y_test)}

Xs, Ys = tinynn.partition(X['train'], Y['train'], size=1000)

# Run training
network = tinynn.Network(DenseBatchNorm(28*28, 100), ReLU(),
                         DenseBatchNorm(100, 10), Softmax())
network.train(Xs, Ys, niter=20,
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
    truth = np.argmax(Y[key], axis=0)
    print("Truth:     ", truth[:30])
    print("Prediction:", labels[:30])

    correct = (truth == labels).mean()
    print("Correct: {:6.2f}%\n".format(100.0 * correct))
