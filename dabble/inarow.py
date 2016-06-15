import collections
import math
import cv2
import scipy.weave
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab


reward_table = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
labels = ['rock', 'paper', 'scissors']


def draw(state, scale=50):
    h, w = state.shape[0] * scale, state.shape[1] * scale
    im = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, state.shape[0]):
        cv2.line(im, (scale / 8, scale * i), (w - scale / 8, scale * i), (255, 255, 255), 1)
    for j in range(1, state.shape[1]):
        cv2.line(im, (scale * j, scale / 8), (scale * j, h - scale / 8), (255, 255, 255), 1)
    for i, j in zip(*np.where(state==1)):
        x, y = scale * j + scale / 4, scale * i + scale / 4
        cv2.line(im, (x, y), (x + scale / 2, y + scale / 2), (255, 255, 255), 1)
        cv2.line(im, (x + scale / 2, y), (x, y + scale / 2), (255, 255, 255), 1)
    for i, j in zip(*np.where(state==2)):
        x, y = scale * j + scale / 2, scale * i + scale / 2
        cv2.circle(im, (x, y), scale / 3, (255, 255, 255), 1)
    return im


def lengths(state):
    '''
    :param state: n x m matrix of (0, 1, 2)
    :return: (max_length of 1s, max_length of 2s) in horizontal,
        vertical or diagonals in state
    '''
    max_length = np.zeros(3, dtype=np.int32)
    code = '''
    #define UPDATE(s) if(s==0) current=0; else for(int k=0;k<=2;++k) if(s==k) {if(current==k) ++current_length;\\
                                                                   else {current=k;current_length=1;};\\
                                                                   max_length(k)=std::max(max_length(k),current_length);\\
                                                                   break;}
    int max_length_1, max_length_2;
    for(int i = 0; i < Nstate[0]; ++i) {
        int current = 0;
        int current_length = 0;
        for(int j=0; j < Nstate[1]; ++j) {
            UPDATE(state(i, j))
        }
    }
    for(int j = 0; j < Nstate[1]; ++j) {
        int current = 0;
        int current_length = 0;
        for(int i=0; i < Nstate[0]; ++i) {
            UPDATE(state(i, j))
        }
    }
    for(int d = -Nstate[0] + 1; d < Nstate[1]; ++d) {
        int current = 0;
        int current_length = 0;
        for(int i = std::max(0, -d); i < std::min(Nstate[0], Nstate[1] - d); ++i) {
            UPDATE(state(i, i + d))
        }
    }
    for(int d = 0; d < Nstate[1] + Nstate[0] - 1; ++d) {
        int current = 0;
        int current_length = 0;
        for(int i = std::max(0, d - Nstate[1] + 1); i < std::min(Nstate[0], d + 1); ++i) {
            UPDATE(state(i, d - i))
        }
    }
    '''
    scipy.weave.inline(
        code, ['state', 'max_length'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return tuple(max_length[1:])


def available_moves(state):
    return zip(*np.where(state==0))


def random_policy(seed=0):
    rng = np.random.RandomState(seed)
    while True:
        yield rng.randint(3)


def avoid_policy(move, seed=0):
    rng = np.random.RandomState(seed)
    options = [i for i in range(3) if i != move]
    while True:
        yield options[rng.randint(2)]


def test_available_moves():
    tests = [([[0, 1, 1, 0], [2, 2, 2, 0]], [(0, 0), (0, 3), (1, 3)]),
             ([[0, 0, 0, 0], [1, 1, 1, 0]], [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)])]
    for m, r in tests:
        assert available_moves(np.array(m, dtype=np.uint8)) == r


def test_lengths():
    def singles(a, b, n):
        ss = []
        for i in range(a):
            for j in range(b):
                s = np.zeros((a, b))
                s[i, j] = n
                ss.append(s)
        return ss
    # empty boards
    assert np.all(np.array([lengths(np.zeros((a,b), dtype=np.uint8))
                            for (a,b) in [(1, 1), (3, 3), (5, 5), (5, 10), (10, 5), (1,7), (8, 1)]])) == 0
    # single-piece boards
    for i in range(2):
        assert np.all(np.array([np.all(np.array([lengths(s)[i] for s in singles(a, b, i+1)]) == 1)
                                for (a,b) in [(1, 1), (3, 3), (5, 5), (5, 10), (10, 5), (1,7), (8, 1)]])) == True
    tests = [([[0, 1, 1, 0], [2, 2, 2, 0]], (2, 3)),
             ([[0, 2, 2, 0], [1, 1, 1, 0]], (3, 2)),
             ([[0, 1, 0, 0], [1, 2, 2, 0]], (2, 2)),
             ([[0, 1, 0, 1], [2, 0, 2, 0]], (1, 1)),
             ([[0, 1, 0, 1], [2, 1, 2, 0]], (2, 1)),
             ([[0, 1, 0, 1], [2, 1, 2, 0], [2, 2, 2, 1]], (2, 3)),
             ([[0, 1, 2, 1], [2, 1, 2, 0], [2, 2, 2, 1]], (2, 3)),
             ([[0, 1, 1, 1], [2, 1, 2, 0], [2, 2, 2, 1]], (3, 3)),
             ([[0, 0, 0, 1], [0, 0, 1, 0], [2, 2, 2, 0]], (2, 3)),
             ([[0, 0, 0, 0], [0, 0, 0, 1], [2, 2, 1, 0]], (2, 2)),
             ([[1, 1, 1, 1], [2, 1, 0, 1], [0, 2, 1, 0]], (4, 2)),
             ([[1, 0, 0, 2], [0, 1, 2, 0], [0, 2, 1, 0], [2, 0, 0, 1]], (4, 4)),
             ]
    for m, r in tests:
        assert lengths(np.array(m, dtype=np.uint8)) == r


if __name__ == "__main__":
    test_lengths()
    test_available_moves()
    im = draw(np.array([[0, 0, 1, 0], [2, 2, 0, 1], [2, 1, 2, 1]], dtype=np.uint8))
    cv2.imshow("kk", im)
    cv2.waitKey(0)