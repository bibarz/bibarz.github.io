import collections
import math
import cv2
import scipy.weave
import numpy as np
import time
import os
import random

class Atan(object):
    @classmethod
    def fwd(cls, x):
        return np.arctan(x)
    @classmethod
    def derivative(cls, x):
        return 1./ (1 + x ** 2.)


class Tanh(object):
    @classmethod
    def fwd(cls, x):
        return np.tanh(x)
    @classmethod
    def derivative(cls, x):
        return 1 - np.tanh(x) ** 2


class Relu(object):
    @classmethod
    def fwd(cls, x):
        return x * (x > 0)
    @classmethod
    def derivative(cls, x):
        return (x > 0).astype(np.float)


class NN(object):
    def __init__(self, sizes, types, init_w_scale = 0.1, init_b_scale = 0.1,
                 eta = 0.1, momentum = 0.):
        assert len(types) == len(sizes) - 1
        self._sizes = sizes
        self._types = types
        self._init_w_scale = init_w_scale
        self._init_b_scale = init_b_scale
        self._eta = eta
        self._momentum = momentum
        self._initialize()

    def _initialize(self):
        self._w = [self._init_w_scale * np.random.randn(self._sizes[i], self._sizes[i + 1]) / np.sqrt(self._sizes[i])
                   for i in range(len(self._sizes) - 1)]
        self._b = [self._init_b_scale * np.random.randn(self._sizes[i + 1])
                   for i in range(len(self._sizes) - 1)]
        self._dw = [np.zeros((self._sizes[i], self._sizes[i + 1]))
                   for i in range(len(self._sizes) - 1)]
        self._db = [np.zeros(self._sizes[i + 1])
                   for i in range(len(self._sizes) - 1)]

    def train(self, batch, teacher):
        assert batch.shape[1:] == tuple(self._sizes[0:1])
        i = []  # layer inputs
        o = [batch]  # layer outputs
        for k in range(len(self._w)):
            i.append(np.dot(o[-1], self._w[k]) + self._b[k])
            o.append(self._types[k].fwd(i[-1]))
        e = teacher - o[-1]
        self._last_gb = []
        self._last_gw = []
        for k in range(len(self._w))[::-1]:
            e *= self._types[k].derivative(i[k])  # backprop the error across the neurons
            self._last_gb = [np.mean(e, axis=0)] + self._last_gb  # prepending, not adding
            self._last_gw = [np.mean(o[k][..., None] * e[:, None, :], axis=0)] + self._last_gw  # prepending, not adding
            e = np.dot(e, self._w[k].T)  # backprop the error across the bridge
        for k, (db, dw) in enumerate(zip(self._last_gb, self._last_gw)):
            self._db[k] = self._momentum * self._db[k] + (1 - self._momentum) * db
            self._dw[k] = self._momentum * self._dw[k] + (1 - self._momentum) * dw
            self._b[k] += self._eta * self._db[k]
            self._w[k] += self._eta * self._dw[k]

    def predict(self, batch):
        assert batch.shape[1:] == tuple(self._sizes[0:1])
        o = batch  # layer outputs
        for k in range(len(self._w)):
            i = np.dot(o, self._w[k]) + self._b[k]
            o = self._types[k].fwd(i)
        return o

    def copy(self):
        c = NN([s for s in self._sizes], [t for t in self._types],
               self._init_w_scale, self._init_b_scale, self._eta, self._momentum)
        c._w = [w.copy() for w in self._w]
        c._b = [b.copy() for b in self._b]
        c._dw = [dw.copy() for dw in self._dw]
        c._db = [db.copy() for db in self._db]
        return c


def test_gradients():
    sizes = [5, 3, 2]
    eps = 1e-9
    n = NN(sizes, [Atan] * (len(sizes) - 1), init_w_scale=0.1)
    w = [x.copy() for x in n._w]
    b = [x.copy() for x in n._b]
    input = np.random.random((1, sizes[0]))
    output = np.ones((1, sizes[-1]))
    e0 = 0.5 * np.sum((n.predict(input) - output) ** 2)
    n.train(input, output)
    gb = [x.copy() for x in n._last_gb]
    gw = [x.copy() for x in n._last_gw]
    for k in range(len(sizes) - 1):
        for i in range(sizes[k]):
            for j in range(sizes[k + 1]):
                w_prime = w[k].copy()
                w_prime[i, j] += eps
                n._w = [x for x in w]
                n._w[k] = w_prime
                n._b = b
                e1 = 0.5 * np.sum((n.predict(input) - output) ** 2)
                assert np.allclose(gw[k][i, j], (e0 - e1) / eps, rtol=1e-3)
        for j in range(sizes[k + 1]):
            b_prime = b[k].copy()
            b_prime[j] += eps
            n._b = [x for x in b]
            n._b[k] = b_prime
            n._w = w
            e1 = 0.5 * np.sum((n.predict(input) - output) ** 2)
            assert np.allclose(gb[k][j], (e0 - e1) / eps, rtol=1e-3)


def test_train():
    # Test that we learn correctly to predict two outputs, each latched to
    # one particular input, with opposite signs
    sizes = [10, 5, 2]
    n_samples = 200
    batch_size = 50
    training_input = np.random.random((n_samples, sizes[0]))
    for j in range(sizes[0]):
        n = NN(sizes, [Tanh] * (len(sizes) - 1), init_w_scale=0.1, eta=0.5, momentum=0.95)
        training_output = np.vstack((0.999 * ((training_input[:, j] > 0.5) * 2 - 1),
                                     0.999 * ((training_input[:, (j + sizes[0] / 2) % sizes[0]] < 0.5) * 2 - 1))).T
        e0 = np.sqrt(np.mean((n.predict(training_input) - training_output) ** 2, axis=0))
        for _ in range(1000):
            for begin in range(0, n_samples, batch_size):
                end = min(n_samples, begin + batch_size)
                n.train(training_input[begin:end], training_output[begin:end])
        e1 = np.sqrt(np.mean((n.predict(training_input) - training_output) ** 2, axis=0))
        # with these parameters (batching and momentum are particularly useful),
        # training should reduce error to about 5% of the initial
        assert np.all(e1 < e0 * 0.1)


def test_copy():
    sizes = [10, 5, 2]
    n_samples = 200
    batch_size = 50
    training_input = np.random.random((n_samples, sizes[0]))

    n = NN(sizes, [Tanh] * (len(sizes) - 1), init_w_scale=0.1, eta=0.5, momentum=0.95)
    training_output = np.vstack((0.999 * ((training_input[:, 0] > 0.5) * 2 - 1),
                                 0.999 * ((training_input[:, sizes[0] / 2] < 0.5) * 2 - 1))).T
    n_out_0 = n.predict(training_input)
    cn = n.copy()
    cn_out_0 = cn.predict(training_input)
    assert np.array_equal(n_out_0, cn_out_0)

    # do some training of the original network
    for _ in range(10):
        for begin in range(0, n_samples, batch_size):
            end = min(n_samples, begin + batch_size)
            n.train(training_input[begin:end], training_output[begin:end])

    # the original network has changed
    n_out_1 = n.predict(training_input)
    assert not np.array_equal(n_out_0, n_out_1)
    # but the copy is intact
    assert np.array_equal(cn_out_0, cn.predict(training_input))

    # Now do the exact same training on the copy
    for _ in range(10):
        for begin in range(0, n_samples, batch_size):
            end = min(n_samples, begin + batch_size)
            cn.train(training_input[begin:end], training_output[begin:end])
    cn_out_1 = cn.predict(training_input)
    # now the output of the copy is the same as the output of the original
    assert np.array_equal(n_out_1, cn_out_1)

    # Train some more and check they still go together
    for _ in range(10):
        for begin in range(0, n_samples, batch_size):
            end = min(n_samples, begin + batch_size)
            n.train(training_input[begin:end], training_output[begin:end])
            cn.train(training_input[begin:end], training_output[begin:end])
            assert np.array_equal(n.predict(training_input), cn.predict(training_input))


if __name__ == "__main__":
    test = True
    if test:
        test_gradients()
        test_train()
        test_copy()
        print "Passed all tests!"
