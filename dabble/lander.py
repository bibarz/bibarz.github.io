from collections import deque
import os
import pickle
import time
import numpy as np
import cv2
import tensorflow as tf


class LanderState(object):
    def __init__(self, l_pos, l_v, base_pos, fuel):
        self.l_pos = l_pos
        self.l_v = l_v
        self.base_pos = base_pos
        self.fuel = fuel


def _to_pix(pos, size):
    return np.round(np.asarray(pos) * size).astype(np.int)


def state_to_im(state, im):
    size = im.shape[::-1]
    im[:] = 0
    lander_size = [0.08, 0.18]
    base_size = [0.28, 0.08]
    pix_pos = _to_pix(state.l_pos, size)
    pix_size = _to_pix(lander_size, size)
    cv2.rectangle(im, tuple(pix_pos - (pix_size[0] / 2, 0)),
                  tuple(pix_pos - (pix_size[0] / 2, 0) + pix_size), 1., -1)
    pix_ground_lt = _to_pix((0., state.base_pos[1]), size)
    pix_ground_rb = _to_pix((1., 0.), size)
    cv2.rectangle(im, tuple(pix_ground_lt),
                  tuple(pix_ground_rb), 0.3, -1)
    pix_pos = _to_pix(state.base_pos, size)
    pix_size = _to_pix(base_size, size)
    cv2.rectangle(im, tuple(pix_pos - pix_size / 2),
                  tuple(pix_pos - pix_size / 2 + pix_size), 0.7, -1)

    return im


def propagate(state, action):
    '''
    action is 0 (thrust to left), 1 (thrust to right), 2 (thrust up), 3 (do nothing)
    '''
    dt = 0.1
    g = np.array([0., -0.5])
    a = {0: np.array([-1., 0.]), 1: np.array([1., 0.]), 2: np.array([0., 1.]), 3: np.array([0., 0.])}[action]
    return LanderState(l_pos=state.l_pos + dt * state.l_v + 0.5 * (a + g) * dt * dt,
                       l_v=np.clip(state.l_v + dt * (a + g), -5, 5),
                       base_pos=state.base_pos,
                       fuel=state.fuel - (action != 3))


def get_action(state):
    '''
    Returns action as 0 (thrust to left), 1 (thrust to right), 2 (thrust up), 3 (do nothing)
    '''
    v_goal = (state.base_pos - state.l_pos) * (1.25, 0.8)
    dv = v_goal - state.l_v
    if np.argmax(np.abs(dv)) == 0:
        a = int((np.sign(dv[0]) + 1) / 2)
    elif dv[1] > 0:
        a = 2
    else:
        a = 3
    return a


def state_to_reward(state):
    if (state.l_pos[1] > 1.5) or (state.l_pos[0] > 1.25) or (state.l_pos[0] < -0.25):  # out of bounds
        return -10.
    if (state.l_pos[1] < state.base_pos[1]):  # crash against ground
        return -10. * np.abs(state.l_v[1])
    if state.fuel <= 0:  # out of fuel
        return -1.
    if np.abs(state.l_pos[1] - state.base_pos[1]) < 0.02 and \
        np.abs(state.l_pos[0] - state.base_pos[0]) < 0.14 and \
        np.abs(state.l_v[0]) < 0.5 and -0.1 < state.l_v[1] <= 0:  # win
        return 20.
    return 0.


def init_state():
    l_pos = np.random.random(2) * [1., 0.2] + [0., 0.8]
    l_v = (np.random.random(2) - 0.5) * [0.02, 0.1]
    base_pos = np.random.random(2) * [0.9, 0.3]
    fuel = 200
    return LanderState(l_pos, l_v, base_pos, fuel)


def play(im_size, n_steps, display=False, player=get_action, state=None):
    '''
    Returns 3 n_steps-length np arrays, and the final state. The three
    arrays are: one with frames, one with actions, one with rewards.
    actions[i] is the action taken when presented with frames[i], and
    rewards[i] is the reward obtained AFTER executing action actions[i],
    i.e., the reward for the state that corresponds to frames[i+1]
    '''
    movie = np.zeros((n_steps, ) + im_size[::-1], dtype=np.float32)
    actions = np.zeros((n_steps, 4), dtype=np.float32)
    if state is None:
        state = init_state()
    reward = np.zeros(n_steps, dtype=np.float32)
    for i in xrange(n_steps):
        state_to_im(state, movie[i])
        a = player(state)
        actions[i, a] = 1.
        state = propagate(state, a)
        r = state_to_reward(state)
        if r < 0.:
            if display:
                print "lose!"
            reward[i] = r
            state = init_state()
        elif r > 0.:
            if display:
                print "win!"
            reward[i] = r
            state = init_state()
        if display:
            cv2.imshow("Lander", cv2.resize(np.flipud(movie[i]), None, None,
                                            160/movie[i].shape[1], 160/movie[i].shape[0], cv2.INTER_NEAREST))
            cv2.waitKey(1)

    return movie, actions, reward, state


def conv_relu(input, kernel_shape, stride):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.truncated_normal_initializer(
        mean=0., stddev=0.01 / np.sqrt(np.prod(kernel_shape[:3]))))
    biases = tf.get_variable("biases", kernel_shape[-1:], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')
#   conv_max = tf.nn.avg_pool(conv, [1, stride, stride, 1], [1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def model(data, prev_outputs, image_size, n_channels, n_actions, n_prev_actions):
    kernel_defs = [(8, 16, 4), (2, 32, 1)]  # each conv layer, (patch_side, n_kernels, stride)
    fc_sizes = [256]
    n_input_kernels = n_channels
    for i, k in enumerate(kernel_defs):
        with tf.variable_scope("conv_%i" % i):
            kernel_shape = (k[0], k[0], n_input_kernels, k[1])
            data = conv_relu(data, kernel_shape, k[2])
            n_input_kernels = k[1]

    for i, n in enumerate(fc_sizes):
        with tf.variable_scope("fc_%i" % i):
            if i == 0:
                previous_n = kernel_defs[-1][1] * np.prod(image_size) / np.prod([k[2] for k in kernel_defs])**2
                data = tf.reshape(data, [-1, previous_n])
                reshape_prev_outputs = tf.reshape(prev_outputs, [-1, n_actions * n_prev_actions])
                prev_outputs_weights = tf.get_variable("prev_outputs_weights", [n_actions * n_prev_actions, n],
                                                       initializer=tf.truncated_normal_initializer(mean=0., stddev=0.01/np.sqrt(n_prev_actions * n_actions)))
            else:
                previous_n = fc_sizes[i-1]
            weights = tf.get_variable("weights", [previous_n, n],
                                      initializer=tf.truncated_normal_initializer(mean=0., stddev=0.01 / np.sqrt(previous_n)))
            biases = tf.get_variable("biases", [n], initializer=tf.constant_initializer(0.0))
            relu_input = tf.matmul(data, weights) + biases
            if i == 0:
                relu_input += 0.1 * (previous_n / n_actions / n_prev_actions) * tf.matmul(reshape_prev_outputs, prev_outputs_weights)
            data = tf.nn.relu(relu_input)

    with tf.variable_scope("flat_out"):
        weights = tf.get_variable("weights", [fc_sizes[-1], n_actions],
                                  initializer=tf.truncated_normal_initializer(mean=0., stddev=0.01 / np.sqrt(fc_sizes[-1])))
        biases = tf.get_variable("biases", [n_actions], initializer=tf.constant_initializer(0.0))
        return tf.matmul(data, weights) + biases


def make_learner(image_size, n_channels, n_actions, n_prev_actions):
    things = {}
    things['graph'] = tf.Graph()
    with things['graph'].as_default():

        # Input and teacher place holders
        things['input'] = tf.placeholder(tf.float32)
        things['output'] = tf.placeholder(tf.float32)
        things['prev_outputs_input'] = tf.placeholder(tf.float32)

        things['logits'] = model(things['input'], things['prev_outputs_input'], image_size, n_channels, n_actions, n_prev_actions)
        things['loss'] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(things['logits'], things['output']))
        things['q_loss'] = tf.reduce_mean(
            tf.square(things['logits'] - things['output']))

        things['learner'] = tf.train.AdamOptimizer(0.001).minimize(things['loss'])
        things['q_learner'] = tf.train.RMSPropOptimizer(1e-9, decay=0.99, epsilon=1e-12).minimize(things['q_loss'])

        # Predictors for the training, validation, and test data.
        things['prediction'] = tf.nn.softmax(things['logits'])
        things['saver'] = tf.train.Saver()

    return things


def make_datasets(movie, actions, reward, n_validation, n_test, frames_per_sequence):
    '''
    Generates training, validation and test datasets from the given movie, actions and reward, returning
        them as three 4-tuples, one with training data one with validation data, one with test data.
    Each 4-tuple is (images, prev_outputs, labels, reward). However, there is a difference between the
        training 4-tuple and the validation and test 4-tuples. The validation and test data is already
        "sequencified", meaning that images[k] is already the set of frames_per_sequence frames to
        feed to the convnet, and prev_outputs[k] is already the set of (frames_per_sequence - 1) outputs
        to feed to the convnet. labels[k] is the action corresponding to the last frame in images[k]. and
        reward[k] is the reward received after performing that action. images, prev_outputs, labels and
        reward have the same length.
    Meanwhile, in the training 4-tuple, images and prev_outputs are not sequencified. This means that
        images[k] is just one frame, so images[k:k+frames_per_sequence] is the sequence of frames at
        whose last frame the player performed the action labels[k] and received the reward reward[k],
        while the actions preceding that action are prev_outputs[k:k+frames_per_sequence - 1]. If N
        is the length of labels and reward, the length of images is N + frames_per_sequence - 1, and
        the length of prev_outputs is N + frames_per_sequence - 2.
    '''
    assert len(movie) >= n_validation + n_test + frames_per_sequence
    n_frames = len(movie) - frames_per_sequence + 1
    n_training = n_frames - n_validation - n_test
    n_actions = actions.shape[1]
    df = frames_per_sequence - 1
    validation_set, validation_prev_outputs, validation_labels, validation_reward = movie[:n_validation + df],\
        actions[:n_validation + df - 1], actions[df:n_validation + df], reward[df:n_validation + df]
    train_set, train_prev_outputs, train_labels, train_reward = movie[n_validation:n_training + n_validation + df],\
        actions[n_validation:n_training + n_validation + df - 1], actions[n_validation + df:n_training + n_validation + df],\
        reward[n_validation + df:n_training + n_validation + df]
    test_set, test_prev_outputs, test_labels, test_reward = movie[n_training + n_validation:],\
        actions[n_validation + n_training:-1], actions[n_training + n_validation + df:], reward[n_training + n_validation + df:]
    assert test_labels.shape[0] == n_test
    assert test_reward.shape[0] == n_test
    assert test_set.shape[0] == n_test + frames_per_sequence - 1

    sequenced_validation = sequencify(validation_set, frames_per_sequence)
    sequenced_validation_prev_outputs = sequencify(validation_prev_outputs, frames_per_sequence - 1)
    sequenced_test = sequencify(test_set, frames_per_sequence)
    sequenced_test_prev_outputs = sequencify(test_prev_outputs, frames_per_sequence - 1)
    assert sequenced_validation.shape[0] == n_validation
    assert sequenced_validation_prev_outputs.shape == (n_validation, n_actions, frames_per_sequence - 1)
    assert sequenced_test.shape[0] == n_test
    assert sequenced_test_prev_outputs.shape == (n_test, n_actions, frames_per_sequence - 1)
    return (train_set, train_prev_outputs, train_labels, train_reward),\
           (sequenced_validation, sequenced_validation_prev_outputs, validation_labels, validation_reward),\
           (sequenced_test, sequenced_test_prev_outputs, test_labels, test_reward)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def sequencify(frames, frames_per_sequence, idx=None, offset=0):
    if idx is None:
        idx = xrange(len(frames) - frames_per_sequence + 1)
    return np.array([np.rollaxis(np.array([frames[j] for j in range(i + offset, i + offset + frames_per_sequence)]),
                                 0, frames[0].ndim + 1) for i in idx])


def display_activators(seqs, predictions):
    winners = np.argmax(predictions, axis=1)
    for i in np.unique(winners):
        activations = predictions[winners==i][:, i]
        weights = activations - np.amin(activations)
        weights /= np.amax(weights)
        weights = weights ** 2
        avg_seq = np.sum(weights * np.rollaxis(seqs[winners==i], 0, seqs.ndim), axis=-1) / np.sum(weights)
        full = np.vstack(np.rollaxis(avg_seq, 2, 0))
        cv2.imshow("%i activator" % i, np.flipud(cv2.resize((full-np.amin(full)) / (np.amax(full) - np.amin(full)),
                                                            None, None, 3., 3.)))
    cv2.waitKey(100)


class Player(object):
    def __init__(self, frames_per_sequence, im_size, mean_im, n_actions):
        self._frame_mem = deque([], maxlen=frames_per_sequence)
        self._action_mem = deque([], maxlen=frames_per_sequence)
        self._im = np.zeros(im_size[::-1], dtype=np.float32)
        self._mean_im = mean_im
        self._n_actions = n_actions

    def play(self, state, things, session, explore_epsilon=0.):
        self._im = state_to_im(state, self._im)
        self._im -= self._mean_im
        self._frame_mem.append(self._im.copy())
        if np.random.random() < explore_epsilon:
            action = np.random.randint(self._n_actions)
        elif len(self._frame_mem) == self._frame_mem.maxlen:
            current_sequence = sequencify(np.asarray(self._frame_mem), len(self._frame_mem))
            current_prev_outputs = sequencify(np.asarray(self._action_mem)[-(len(self._frame_mem) - 1):],
                                              len(self._frame_mem) - 1)
            feed_dict = {things['input']: current_sequence, things['prev_outputs_input']: current_prev_outputs}
            prediction = session.run(things['prediction'], feed_dict=feed_dict)
            assert prediction.shape == (1, self._n_actions)
            action = np.argmax(prediction[0])
        else:
            action = np.random.randint(self._n_actions)
        self._action_mem.append(np.eye(self._n_actions, dtype=np.float32)[action])
        return action


def load_model(folder, instance_filename):
    with open(os.path.join(folder, 'model.pickle'), 'r') as f:
        model_dict = pickle.load(f)
    things = make_learner(model_dict['frame_size'],
                          model_dict['frames_per_sequence'],
                          model_dict['n_actions'],
                          model_dict['frames_per_sequence'] - 1)
    return things, model_dict


def learn_supervised(save=True):
    frame_size = (32, 32)
    frames_per_sequence = 6
    n_training = 50000
    n_validation = 5000
    n_test = 5000
    n_passes = 4
    batch_size = 64
    n_frames = n_training + n_validation + n_test
    movie, actions, reward, _ = play(frame_size, n_frames + frames_per_sequence - 1)
    print "Dataset win: %i, lose: %i" % (np.sum(reward > 0), np.sum(reward < 0))
    mean_im = np.mean(movie, axis=0)
    movie -= mean_im
    n_actions = actions.shape[1]
    things = make_learner(frame_size, frames_per_sequence, n_actions, frames_per_sequence - 1)

    n_steps = n_training * n_passes / batch_size
    t0_s=time.strftime("%c")
    folder = '/home/ubuntu/lander/' + t0_s
    os.mkdir(folder)
    model_dict = dict(frame_size=frame_size, frames_per_sequence=frames_per_sequence,
                      n_actions=n_actions, mean_im=mean_im)
    with open(os.path.join(folder, 'model.pickle'), 'w') as f:
        pickle.dump(model_dict, f)

    with tf.Session(graph=things['graph']) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        train, validation, test = make_datasets(movie, actions, reward, n_validation, n_test, frames_per_sequence)
        train_idx = np.random.permutation(n_training)
        for step in range(n_steps):
            offset = (step * batch_size) % (n_training - batch_size)
            idx = train_idx[offset:(offset + batch_size)]
            batch_data = sequencify(train[0], frames_per_sequence, idx)
            batch_prev_outputs = sequencify(train[1], frames_per_sequence - 1, idx)
            batch_labels = train[2][idx]
            feed_dict = {things['input']: batch_data, things['prev_outputs_input']: batch_prev_outputs, things['output']: batch_labels}
            session.run(things['learner'], feed_dict=feed_dict)
            if (step % (n_steps / 10) == 0):
                l, predictions = session.run(
                    [things['loss'], things['prediction']], feed_dict=feed_dict)
                print('Minibatch loss at step %d/%d: %.3f, accuracy %.1f%%' % (step, n_steps, l, accuracy(predictions, batch_labels)))
                feed_dict = {things['input']: validation[0], things['prev_outputs_input']: validation[1], things['output']: validation[2]}
                l, predictions, = session.run([things['loss'], things['prediction']], feed_dict=feed_dict)
                print('Validation loss %.3f, accuracy: %.1f%%' % (l, accuracy(predictions, validation[2])))
                display_activators(validation[0], predictions)
                if save:
                    things['saver'].save(session, os.path.join(folder, 'instance_%.5i' % step))

        feed_dict = {things['input']: test[0], things['prev_outputs_input']: test[1]}
        predictions, = session.run([things['prediction']], feed_dict=feed_dict)
        print('Test accuracy: %.1f%%' % accuracy(predictions, test[2]))
        player = Player(frames_per_sequence, frame_size, mean_im, n_actions)

        print "Let's play"
        _, _, reward, _ = play(frame_size, 2000, display=True, player=lambda state: player.play(state, things, session))
        print "Win: %i, lose: %i" % (np.sum(reward > 0), np.sum(reward < 0))


def load_and_play(folder, instance_filename):
    things, model_dict = load_model(folder, instance_filename)
    with tf.Session(graph=things['graph']) as session:
        things['saver'].restore(session, os.path.join(folder, instance_filename))
        player = Player(model_dict['frames_per_sequence'], model_dict['frame_size'], model_dict['mean_im'], model_dict['n_actions'])

        print "Let's play"
        _, _, reward, _ = play(model_dict['frame_size'], 2000, display=True, player=lambda state: player.play(state, things, session))
        print "Win: %i, lose: %i" % (np.sum(reward > 0), np.sum(reward < 0))


def load_and_reinforce(folder, instance_filename):
    things, model_dict = load_model(folder, instance_filename)
    frames_per_sequence = model_dict['frames_per_sequence']
    frame_size = model_dict['frame_size']
    n_actions = model_dict['n_actions']
    mean_im = model_dict['mean_im']
    mem_size = 100000
    frame_mem = deque([], mem_size)
    action_mem = deque([], mem_size)
    reward_mem = deque([], mem_size)
    n_validation = 500
    replay_size = 31
    gamma = 0.95
    explore_epsilon = 0.1
    n_training = 500000
    n_training_per_epoch = 1000
    state = None
    player = Player(frames_per_sequence, frame_size, mean_im, n_actions)
    with tf.Session(graph=things['graph']) as session:
        things['saver'].restore(session, os.path.join(folder, instance_filename))
        tf.get_variable_scope().reuse_variables()
        w = tf.get_variable("flat_out/weights")
        b = tf.get_variable("flat_out/biases")
        w.assign(w / 1000.).op.run()
        b.assign(b / 1000.).op.run()
        print('Initialized')
        movie, actions, reward, _ = play(frame_size, 10 + n_validation + 10 + frames_per_sequence - 1, player=lambda state:player.play(state, things, session))
        _, validation, _ = make_datasets(movie, actions, reward, n_validation, 10, frames_per_sequence)
        for iteration in range(n_training):
            movie, actions, reward, state = play(frame_size, 1,
                player=lambda state:player.play(state, things, session, explore_epsilon), state=state, display=True)
            frame_mem.append(movie[0])
            action_mem.append(actions[0])
            reward_mem.append(reward[0])
            if len(frame_mem) < 500 * replay_size + frames_per_sequence + 1:
                continue
            idx = np.random.randint(0, len(frame_mem) - frames_per_sequence - 1, replay_size)
#            idx[-1] = len(frame_mem) - frames_per_sequence - 1
            batch_data = sequencify(frame_mem, frames_per_sequence, idx)
            batch_prev_outputs = sequencify(action_mem, frames_per_sequence - 1, idx)
            batch_next_data = sequencify(frame_mem, frames_per_sequence, idx, offset=1)
            batch_next_prev_outputs = sequencify(action_mem, frames_per_sequence - 1, idx, offset=1)
            batch_actions = np.array([action_mem[i] for i in idx + frames_per_sequence - 1])
            batch_rewards = np.array([reward_mem[i] for i in idx + frames_per_sequence - 1])
            feed_dict = {things['input']: batch_data, things['prev_outputs_input']: batch_prev_outputs}
            current_q = session.run(things['logits'], feed_dict=feed_dict)  # get the current Q values
            feed_dict_next = {things['input']: batch_next_data, things['prev_outputs_input']: batch_next_prev_outputs}
            next_q = session.run(things['logits'], feed_dict=feed_dict_next)  # get the next Q values
            next_values = (batch_rewards == 0) * gamma * np.amax(next_q, axis=1) + batch_rewards  # only non-terminal (reward=0) get Q values as part of the value
            current_q[np.arange(current_q.shape[0]), np.argmax(batch_actions, axis=1)] = next_values
            feed_dict = {things['input']: batch_data, things['prev_outputs_input']: batch_prev_outputs, things['output']: current_q}
            session.run(things['q_learner'], feed_dict=feed_dict)
            if iteration % n_training_per_epoch == 0:
                epoch = iteration / n_training_per_epoch
                l, modified_q = session.run(
                    [things['q_loss'], things['logits']], feed_dict=feed_dict)
                print('Epoch %i q_loss, q_avg: %.4f, %.4f' % (epoch, l, np.mean(np.amax(modified_q, axis=1))))
                feed_dict = {things['input']: validation[0], things['prev_outputs_input']: validation[1]}
                q = session.run(things['logits'], feed_dict=feed_dict)
                print('Validation q_avg: %.6f' % np.mean(np.amax(q, axis=1)))
                epoch_reward = np.array([reward_mem[i] for i in range(len(reward_mem) - n_training_per_epoch, len(reward_mem))])
                print "Epoch win: %i, lose: %i" % (np.sum(epoch_reward > 0), np.sum(epoch_reward < 0))

                # if epoch % 100 == 0:
                #     things['saver'].save(session, os.path.join(folder, instance_filename, '_reinforce_epoch_%.2i' % epoch))

        feed_dict = {things['input']: test[0], things['prev_outputs_input']: test[1]}
        q = session.run(things['logits'], feed_dict=feed_dict)
        print('Test q_avg: %.2f' % np.mean(np.amax(q, axis=1)))

        print "Let's play"
        _, _, reward, _ = play(frame_size, 2000, display=True, player=lambda state: player.play(state, things, session))
        print "Win: %i, lose: %i" % (np.sum(reward > 0), np.sum(reward < 0))



def gsum(values, shifts):
    total = []
    real_lengths = [len(v) + s for v, s in zip(values, shifts)]
    max_length = max(real_lengths)
    carry = 0
    i = 0
    while carry != 0 or i < max_length:
        for j in range(len(values)):
            if i >= shifts[j] and i < real_lengths[j]:
                carry += values[j][i - shifts[j]]
        total.append(carry % 10)
        carry /= 10
        i += 1
    return total


def gprod(l1, l2):
    n1 = len(l1)
    n2 = len(l2)
    if n1 == 0 or n2 == 0:
        return []
    if n1 == 1 and n1 == 1:
        m = l1[0] * l2[0]
        if m >= 10:
            return [m % 10, m / 10]
        else:
            return [m]
    else:
        a, b = l1[n1 / 2:], l1[:n1 / 2]
        c, d = l2[n2 / 2:], l2[:n2 / 2]
        return gsum([gprod(a, c), gprod(b, c), gprod(a, d), gprod(b, d)], [n1 / 2 + n2 / 2, n2 / 2, n1 / 2, 0])


if __name__ == "__main__":
#    learn_supervised()
#    load_and_play('/home/ubuntu/lander/Thu Mar 31 10:49:03 2016', 'instance_12500')
    load_and_reinforce('/home/ubuntu/lander/Thu Mar 31 11:51:34 2016', 'instance_03120')
#    play((80, 80), 100000, display=False)
#    print gprod([3,5], [2,4])
