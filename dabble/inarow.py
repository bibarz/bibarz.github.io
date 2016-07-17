import collections
import math
import cv2
import scipy.weave
import numpy as np
import time
import os
import random
import tensorflow as tf
from matplotlib import pylab
from nn import NN, Tanh


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


def hard_reward(state, L):
    l1, l2 = lengths(state)
    w1 = l1 >= L
    w2 = l2 >= L
    if w1 == w2:
        return 0.
    if w1:
        return 1.
    return -1.


def is_terminal(state, L):
    return np.all(state != 0) or max(lengths(state)) >= L



class ValueNet(object):
    def __init__(self, shape, layer_sizes = []):
        self._shape = shape
        all_sizes = [np.prod(shape)] + layer_sizes + [1]
        self._n = NN(all_sizes, [Tanh] * len(all_sizes) - 1, eta=0.1, momentum=0.)

    def _reshape(self, data):
        '''
        Turn a batch of 2d (0, 1, 2) game states into a batch of 1d (-1., 0., 1.) states
        '''
        return np.reshape(data - 3 * (data > 1.5), (data.shape[0], -1)).astype(np.float)

    def get_values(self, states):
        return self._n.predict(self._reshape(states))

    def learn_values(self, states, values):
        self._n.train(self._reshape(states), values)


class Game(object):
    def __init__(self, shape, L, player_1, player_2):
        self._L = L
        self._players = [player_1, player_2]
        self._turn = 1

    def play(self, state):
        move = self._players[self._turn - 1].play(state)
        assert move in available_moves(state)
        state[move[0], move[1]] = self._turn
        self._turn = 3 - self._turn
        return state


class InterfacePlayer(object):
    def __init__(self, name, number):
        self._name = name
        assert number in [1, 2]
        self._number = number
        self._scale = 50
        self._regions = None

    def play(self, state):
        self._move = None
        self._state = state
        cv2.namedWindow(self._name)
        im, self._regions = draw(state)
        cv2.imshow(self._name, im)
        cv2.setMouseCallback(self._name, self._on_mouse)
        cv2.waitKey(10)
        while self._move is None:
            cv2.waitKey(50)
        return self._move

    def _on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP:
            assert self._regions is not None
            for r, c in self._regions:
                if y > r[0] and y < r[1] and x > r[2] and x < r[3]:
                    if self._state[c[0], c[1]] == 0:
                        self._move = c


def length_terminal_value(state):
    l = lengths(state)
    return (l[0] - l[1]) / 1000.


class AlphaBetaPlayer(object):
    def __init__(self, name, number, L, depth, gamma, terminal_value_function, use_alphabeta=True):
        self._name = name
        assert number in [1, 2]
        self._number = number
        self._depth = depth
        self._L = L
        self._gamma = gamma
        self._terminal_value_function = terminal_value_function
        self._use_alphabeta = use_alphabeta

    def play(self, state):
        sign = [1, -1][self._number - 1]  # +1 for player 1, -1 for player 2
        v, a = self._minmax_alphabeta(state, self._depth, -1e9, 1e9, sign)
        print("player %i thinks move to %s will produce value %.4f" % (self._number, a, v))
        return a

    def _minmax_alphabeta(self, state, depth, alpha, beta, sign):
        '''
        alphabeta is [alpha, beta]
        sign=1 for max and player 1, -1 for min and player 2
        '''
        if is_terminal(state, self._L):
            return hard_reward(state, self._L), None
        if depth == 0:
            v = self._terminal_value_function(state)
            return v, None
        max_v = -1e9
        best_a = None
        moves = available_moves(state)
        alphabeta = [alpha, beta]
        alphabeta_to_pass = [alpha, beta]  # we must wipe out beta (max) or alpha (min) when we pass it down to the next layer
        alphabeta_to_pass[sign > 0] = sign * 1e9
        for a in moves:
            new_state = state.copy()
            new_state[a[0], a[1]] = 1 + (sign < 0)
            v, _ = self._minmax_alphabeta(new_state, depth - 1, alphabeta_to_pass[0], alphabeta_to_pass[1], -sign)
            v *= (sign * self._gamma)
            if v > max_v:
                max_v = v
                best_a = a
            if self._use_alphabeta and v > sign * alphabeta[sign > 0]:
                break
            if self._use_alphabeta and v > sign * alphabeta_to_pass[sign < 0]:
                alphabeta_to_pass[sign < 0] = sign * v
        return sign * max_v, best_a


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
    regions = [((i * scale, (i + 1) * scale, j * scale, (j + 1) * scale), (i, j))
               for j in range(state.shape[1]) for i in range(state.shape[0])]
    return im, regions


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


def test_alphabeta():
        L = 3
        shape = (3, 4)
        starts = [(7, ((0, 0), (1, 1))), (5, ((0, 0), (1, 0))),
                  (5, ((1, 1), (0, 0))), (5, ((1, 1), (1, 2))), (5, (None, None))]
        for d, s in starts:
            p1 = AlphaBetaPlayer("p1", 1, L, d, 0.99, length_terminal_value, use_alphabeta=True)
            p1_raw = AlphaBetaPlayer("p1", 1, L, d, 0.99, length_terminal_value, use_alphabeta=False)
            p2 = AlphaBetaPlayer("p2", 2, L, d, 0.99, length_terminal_value, use_alphabeta=True)
            p2_raw = AlphaBetaPlayer("p2", 2, L, d, 0.99, length_terminal_value, use_alphabeta=False)
            state = np.zeros(shape)
            if s[0] is not None:
                state[s[0][0], s[0][1]] = 1
                state[s[1][0], s[1][1]] = 2
            state_raw = state.copy()
            game = Game(shape, L, p1, p2)
            game_raw = Game(shape, L, p1_raw, p2_raw)
            while not is_terminal(state, L):
                state = game.play(state)
                state_raw = game_raw.play(state_raw)
                assert np.array_equal(state, state_raw)


if __name__ == "__main__":
    test = False
    if test:
        test_alphabeta()
        test_lengths()
        test_available_moves()
    else:
        L = 3
        shape = (3, 4)
        state = np.zeros(shape)
        p1 = InterfacePlayer("p1", 1)
        p2 = AlphaBetaPlayer("p2", 2, L, 12, 0.99, length_terminal_value, use_alphabeta=True)
        game = Game(shape, L, p1, p2)
        while not is_terminal(state, L):
            state = game.play(state)
        print hard_reward(state, L)
