import cv2
import scipy.weave
import numpy as np
from nn import NN, Tanh, Relu


def all_states(shape):
    l = np.prod(shape)
    all_states = (np.arange(3 ** l)[:, None] / (3 ** np.arange(l))[None, :]) % 3
    diff = np.sum(all_states == 1, axis=1) - np.sum(all_states == 2, axis=1)
    good_states = (diff == 0) | (diff == 1)  # either player one has one more move or both players have the same
    all_states = all_states[good_states].reshape((-1, ) + shape)
    return all_states


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
    def __init__(self, shape, layer_sizes=[], batch_size=50, eta=0.1, momentum=0.9,
                 init_w_scale=0.1, init_b_scale=0.1):
        self._shape = shape
        all_sizes = [np.prod(shape) * 3] + layer_sizes + [1]
        self._n = NN(all_sizes, [Relu] * (len(all_sizes) - 2) + [Tanh], eta=eta, momentum=momentum,
                     init_w_scale=init_w_scale, init_b_scale=init_b_scale)
        self._batch_size = batch_size
        self._batch_states = np.zeros((self._batch_size, all_sizes[0]))
        self._batch_values = np.zeros((self._batch_size, 1))
        self._batch_idx = 0

    def copy(self):
        c = ValueNet(self._shape, [], self._batch_size)
        c._n = self._n.copy()
        c._batch_states = self._batch_states.copy()
        c._batch_values = self._batch_values.copy()
        c._batch_idx = self._batch_idx
        return c

    def _reshape_batch(self, data):
        '''
        Turn a batch of 2d (0, 1, 2) game states into a batch of one-hot, 1d vectors
            where each square in the original game board is 3 values,
            (1, 0, 0) for empty, (0, 1, 0) for player 1, and (0, 0, 1) for player 2
        '''
        reshaped_data = np.reshape(data, (data.shape[0], -1))
        output = np.zeros((data.shape[0], 3 * np.prod(data.shape[1:])))
        code = '''
            for(int i = 0; i < Nreshaped_data[0]; ++i) {
                for(int j = 0; j < Nreshaped_data[1]; ++j) {
                    output(i, 3 * j + (int)reshaped_data(i, j)) = 1;
                }
            }
        '''
        scipy.weave.inline(
            code, ['reshaped_data', 'output'],
            type_converters=scipy.weave.converters.blitz, compiler='gcc',
            extra_compile_args=["-O3"]
        )
        return output

    def _reshape_single(self, state):
        '''
        Turn a single 2d (0, 1, 2) state into a single 1d (-1., 0., 1.) state
        '''
        return self._reshape_batch(np.array([state]))[0]

    def get_values(self, states):
        return self._n.predict(self._reshape_batch(states))

    def get_single_value(self, state):
        return self.get_values(np.array([state]))[0]

    def learn_values(self, states, values):
        self._n.train(self._reshape_batch(states), values)

    def learn_single_value(self, state, value):
        '''
        Will not learn immediately, but only when batch_size samples
        have been gathered
        '''
        self._batch_states[self._batch_idx] = self._reshape_single(state)
        self._batch_values[self._batch_idx] = value
        self._batch_idx += 1
        if self._batch_idx == self._batch_size:
            self._n.train(self._batch_states, self._batch_values)
            self._batch_idx = 0


class Game(object):
    def __init__(self, shape, L, player_1, player_2, debug=True):
        self._L = L
        self._players = [player_1, player_2]
        self._turn = 1
        self._debug = debug

    def play(self, state):
        move = self._players[self._turn - 1].play(state)
        if self._debug:
            assert move in available_moves(state)
        state[move[0], move[1]] = self._turn
        self._turn = 3 - self._turn
        return state

    def reset(self):
        self._turn = 1

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
    def __init__(self, name, number, L, depth, gamma,
                 terminal_value_function, use_alphabeta=True, learner=None, rng=None):
        self._name = name
        assert number in [1, 2]
        self._number = number
        self._depth = depth
        self._L = L
        self._gamma = gamma
        self._terminal_value_function = terminal_value_function
        self._use_alphabeta = use_alphabeta
        self._learner = learner
        self.last_v = None
        if rng is None:
            rng = np.random.RandomState()
        self._rng = rng

    def play(self, state):
        sign = [1, -1][self._number - 1]  # +1 for player 1, -1 for player 2
        v, a = self._minmax_alphabeta(state, self._depth, 1e9, sign)
        if self._learner is not None:
            self._learner.learn_single_value(state, v)
        self.last_v = v
        return a

    def get_value(self, state):
        self.play(state)
        return self.last_v

    def _minmax_alphabeta(self, state, depth, alpha, sign):
        '''
        sign=1 for max and player 1, -1 for min and player 2
        alpha plays the role of beta if sign is 1 or alpha if sign is -1
        '''
        if is_terminal(state, self._L):
            return hard_reward(state, self._L), None
        if depth == 0:
            return self._terminal_value_function(state), None
        max_v = -1e9
        best_a = [None]
        for a in available_moves(state):
            new_state = state.copy()
            new_state[a[0], a[1]] = 1 + (sign < 0)
            v, _ = self._minmax_alphabeta(new_state, depth - 1, -max_v, -sign)
            v *= self._gamma
            assert abs(v) < 1e8  # should never happen that the return value is uninitialized
            if self._learner is not None:
                self._learner.learn_single_value(new_state, v)
            v *= sign
            if v >= max_v:
                if v == max_v:
                    best_a.append(a)
                else:
                    max_v = v
                    best_a = [a]
            if self._use_alphabeta and v > alpha:
                break
        return sign * max_v, best_a[self._rng.randint(len(best_a))]


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


def self_play():
    shape = (3, 3)
    L = 3
    all_s = all_states(shape)
    all_1 = np.sum(all_s == 1, axis=(1, 2))
    all_2 = np.sum(all_s == 2, axis=(1, 2))
    assert np.all((all_1 - all_2 == 1) | (all_1 == all_2))
    p1_s = all_s[all_1 == all_2]
    p2_s = all_s[all_1 > all_2]
    p1 = AlphaBetaPlayer("p1", 1, L, np.prod(shape), 0.99, length_terminal_value, use_alphabeta=True)
    p2 = AlphaBetaPlayer("p2", 2, L, np.prod(shape), 0.99, length_terminal_value, use_alphabeta=True)
    v1 = [p1.get_value(s) for s in p1_s]
    v2 = [p2.get_value(s) for s in p2_s]
    all_s = np.concatenate((p1_s, p2_s), axis=0)
    all_v = np.hstack((v1, v2))[:, None]
    positives = all_v > 0
    negatives = all_v < 0
    zeros = all_v == 0
    print("avg_pos: %.3f, avg_neg: %.3f, avg_zero: %.3f" %
          (np.mean(all_v[positives]), np.mean(all_v[negatives]),
           np.mean(all_v[zeros])))

    golden_starts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    golden_states = [np.zeros(shape) for _ in golden_starts]
    for s, (i, j) in zip(golden_states, golden_starts):
        s[i, j] = 1.
    golden_indices = [np.where(np.all(all_s == g, axis=(1,2)))[0][0] for g in golden_states]

    v = ValueNet(shape, [25, 25], eta=0.02, momentum=0.8, init_w_scale=0.1, init_b_scale=0.1)
    batch_size = 256
    import cPickle
    for iteration in range(100000):
        order = np.random.permutation(len(all_s))
        for i in range(len(all_s) / batch_size + 1):
            v.learn_values(all_s[order[i * batch_size:(i + 1) * batch_size]], all_v[order[i * batch_size:(i + 1) * batch_size]])
        if iteration % 500 == 0:
            current_values = [v.get_single_value(s) for s in golden_states]
            all_current_values = v.get_values(all_s)
            rms = np.sqrt(np.mean((all_current_values - all_v) ** 2))
            print("rms: %.3f, avg_pos: %.3f, avg_neg: %.3f, avg_zero: %.3f, current: %s; golden: %s" %
                  (rms, np.mean(all_current_values[positives]), np.mean(all_current_values[negatives]),
                   np.mean(all_current_values[zeros]), current_values, all_v[golden_indices, 0]))
            with open('valuenet_%s.pkl'%(shape, ), 'w') as f:
                cPickle.dump(v, f)
    assert False

    L = 3
    gamma = 0.99
    shape = (3, 3)

    golden_starts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    golden_states = [np.zeros(shape) for _ in golden_starts]
    for s, (i, j) in zip(golden_states, golden_starts):
        s[i, j] = 1.
#    golden_values = [0.92274469442792, 0.9414801494009999, -0.9135172474836407, 0.9414801494009999]
    golden_values = []
    golden_player = AlphaBetaPlayer("golden", 2, L, 11, gamma, length_terminal_value, use_alphabeta=True)
    for s in golden_states:
        _ = golden_player.play(s)
        golden_values.append(golden_player.last_v)
    print("Golden values: %s" % golden_values)

    v = ValueNet(shape, [20], eta=0.025)
    for round in range(5000):
        if round % 50 == 0:
            frozen_v = v.copy()
            if (round % 100 == 0):
                p1 = AlphaBetaPlayer("p1", 1, L, 3, gamma, v.get_single_value, use_alphabeta=False, learner=v, rng=np.random.RandomState(0))
                p2 = AlphaBetaPlayer("p2", 2, L, 3, gamma, frozen_v.get_single_value, use_alphabeta=True, rng=np.random.RandomState(0))
            else:
                p1 = AlphaBetaPlayer("p1", 1, L, 3, gamma, frozen_v.get_single_value, use_alphabeta=True, rng=np.random.RandomState(0))
                p2 = AlphaBetaPlayer("p2", 2, L, 3, gamma, v.get_single_value, use_alphabeta=False, learner=v, rng=np.random.RandomState(0))
            game = Game(shape, L, p1, p2, debug=False)
        state = np.zeros(shape)
        game.reset()
        while not is_terminal(state, L):
            state = game.play(state)
        if (round % 50 == 0):
            current_values = [v.get_single_value(s) for s in golden_states]
            print("current: %s; golden: %s" % (current_values, golden_values))


if __name__ == "__main__":
    test = False
    if test:
        test_alphabeta()
        test_lengths()
        test_available_moves()
        print "Passed all tests!"
    else:
        self_play()
        # L = 3
        # shape = (3, 4)
        # state = np.zeros(shape)
        # p1 = InterfacePlayer("p1", 1)
        # p2 = AlphaBetaPlayer("p2", 2, L, 12, 0.99, length_terminal_value, use_alphabeta=True)
        # game = Game(shape, L, p1, p2)
        # while not is_terminal(state, L):
        #     state = game.play(state)
        # print hard_reward(state, L)
