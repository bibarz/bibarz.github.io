import os
import numpy as np
import tensorflow as tf
import zipfile
import pickle
import stat
from six.moves.urllib.request import urlretrieve
import time
import cv2
import sys


def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    f = zipfile.ZipFile(filename)
    l = f.namelist()
    assert len(l) == 1
    name = l[0]
    data = tf.compat.as_str(f.read(name))
    f.close()
    return data


def compile_data(folder):
    '''
    Puts together all txt files in a folder
    '''
    filenames = [f for f in os.listdir(folder) if os.path.splitext(f)[1][-3:] == 'txt']
    data = ''
    for fn in filenames:
        with open(os.path.join(folder, fn), 'r') as f:
            data += tf.compat.as_str(f.read())
    return data


class Encoder(object):
    def __init__(self, data, default_char=ord(' '), min_freq=1e-4):
        '''
        data should be a uint8 np array from a string. From it we deduce
        the existing characters, and the very unusual ones (frequency less
        than min_freq) get mapped to default_char (which should be in the data)
        '''
        chars, idx = np.unique(data, return_inverse=True)
        counts = np.bincount(idx)
        freqs = counts / np.sum(counts).astype(np.float)
        order = np.argsort(freqs)[::-1]
        chars, counts, freqs = chars[order], counts[order], freqs[order]
        unfreq = freqs < min_freq
        assert default_char in chars[~unfreq]
        self._decoder = chars[~unfreq]
        self._encoder = np.zeros(np.amax(chars) + 1, dtype=chars.dtype)
        self._encoder[chars[~unfreq]] = np.arange(np.sum(~unfreq))
        self._encoder[chars[unfreq]] = self._encoder[default_char]
        self.vocabulary_size = len(self._decoder)

    def char2id(self, chars):
        return self._encoder[chars]

    def id2char(self, ids):
        return self._decoder[ids]

    def print_chars(self):
        for i, c in enumerate(self._decoder):
            print i, " --- ", c, " --- ", c.tostring().decode('mac-roman')


def batch_generator(text, batch_size, num_unrollings, encoder, unroll_shift):
    '''
    generates one-hot encoded batches from the text
    '''
    segment = len(text) // batch_size
    cursor = np.arange(num_unrollings + 1, dtype=np.int)[:, None] +\
        segment * np.arange(batch_size, dtype=np.int)[None, :]
    unrolled_batch = np.zeros((num_unrollings + 1, batch_size, encoder.vocabulary_size), dtype=np.float)
    idx1, idx2 = np.indices(unrolled_batch.shape[:2])
    while True:
        unrolled_batch[:] = 0
        unrolled_batch[idx1, idx2, encoder.char2id(text[cursor % len(text)])] = 1.0
        cursor += unroll_shift + 1
        yield unrolled_batch


def characters(probabilities, encoder):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return encoder.id2char(np.argmax(probabilities, -1))


def print_batches(batches, encoder):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    str_batch = [s.tostring() for s in characters(batches, encoder).T]
    for s in str_batch:
        print s.decode('mac-roman'), ':::',
    print


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample(prediction, temperature, n=1, repeat=True):
    """
    Turn row predictions into n-hot encoded samples.
    Note than n > 1 results in disaster (the network does
    not "generalize" from inputs consisting of 1 character
    to inputs consisting of some distribution of probs over
    characters)
    """
    assert temperature >= 0
    temperature = max(temperature, 1e-9)
    log_prob = np.log(np.maximum(prediction, 1e-9))
    log_prob /= temperature
    prediction = np.exp(log_prob)
    prediction /= np.sum(prediction, axis=1)
    p = np.zeros_like(prediction)
    cum_prob = np.cumsum(prediction, axis=1)
    for i in range(n):
        values = np.random.random(prediction.shape[0])
        idx = np.argmax(values[:, None] <= cum_prob, axis=1)
        p[np.arange(p.shape[0]), idx] = 1.0
        if not repeat:
            prediction[np.arange(p.shape[0]), idx] = 0.0
            prediction /= np.sum(prediction, axis=1)
            cum_prob = np.cumsum(prediction, axis=1)
    return p / np.sum(p, axis=1)


def random_distribution(vocabulary_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:,None]


def display_samples(samples, encoder):
    n_to_show = 20
    im = np.zeros(((n_to_show + 1) * 20, 500, 3), dtype=np.uint8)
    logprobs = [np.sum(s[1]) for s in samples]
    order = np.argsort(logprobs)[::-1]
    for i, idx in enumerate(order[:n_to_show]):
        s = samples[idx]
        txt = ("%.2f " % (logprobs[idx])) + encoder.id2char(np.array(s[0], dtype=np.int)).tostring()
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]
        cv2.putText(im, txt, (20, 20 * (i + 1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255))
        next_chars = encoder.id2char(np.argsort(s[2][0])[::-1][:3]).tostring()  # the three most likely next characters
        cv2.putText(im, next_chars, (20 + size[0], 20 * (i + 1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
    cv2.imshow("Samples", im)
    cv2.waitKey(1)


def text_generator(session, things, encoder, seed, temperature, depth=1, n_samples=1):
    eval_feed_dict = {v: np.zeros(v._shape) for v in [things['sampler_combined_start']]}
    eval_feed_dict[things['dropout_keep_prob']] = 1.0
    run_list = [things['sampler_combined_end'], things['sampler_prediction']]
    for s in seed:
        feed = np.zeros((1, encoder.vocabulary_size), dtype=np.float32)
        feed[0, s] = 1.
        yield characters(feed, encoder).T[0].tostring().decode('mac-roman')
        eval_feed_dict[things['sampler_input']] = feed
        eval_feed_dict[things['sampler_combined_start']], predictions = session.run(run_list, feed_dict=eval_feed_dict)

    samples = [((), (0., ), predictions, eval_feed_dict[things['sampler_combined_start']])]
    current_depth = 0
    while True:
        all_p = np.hstack([np.sum(c[1]) + c[2] for c in samples])
        all_p = np.exp(all_p)
        all_p /= np.sum(all_p, axis=1)
        first_selected = np.where(sample(all_p, temperature))[1]
        if (current_depth >= depth):  # produce the best character
            feed = np.zeros((1, encoder.vocabulary_size), dtype=np.float32)
            first_selected_sample = first_selected / encoder.vocabulary_size
            c = samples[first_selected_sample][0][0]
            feed[0, c] = 1.  # the first character of the character list (index 0) of the first chosen sample
            yield characters(feed, encoder).T[0].tostring().decode('mac-roman')
            samples = [s for s in samples if s[0][0] == c]  # we have to eliminate all samples that don't start with the chosen one
            all_p = np.hstack([np.sum(c[1]) + c[2] for c in samples])
            all_p = np.exp(all_p)
            all_p /= np.sum(all_p, axis=1)
            selected = []
            n_draw = n_samples
        else:
            selected = [first_selected]
            all_p[0, first_selected] = 0
            n_draw = n_samples - 1
        for i in range(min(n_draw, all_p.shape[1])):
            selected.append(np.where(sample(all_p, temperature))[1])
            all_p[0, selected[-1]] = 0  # do not repeat choice
        selected = np.array(selected)
        selected_samples = selected / encoder.vocabulary_size
        selected_characters = selected % encoder.vocabulary_size
        new_samples = []
        for s, c in zip(selected_samples, selected_characters):
            current_sample = samples[s]
            eval_feed_dict[things['sampler_combined_start']] = current_sample[3]
            feed = np.zeros((1, encoder.vocabulary_size), dtype=np.float32)
            feed[0, c] = 1.
            eval_feed_dict[things['sampler_input']] = feed
            new_state, predictions = session.run(run_list, feed_dict=eval_feed_dict)
            if current_depth >= depth:
                ns = (current_sample[0][1:] + (c,), current_sample[1][1:] + (np.log(current_sample[2][0, c]), ),
                      predictions, new_state)
            else:
                ns = (current_sample[0] + (c,), current_sample[1] + (np.log(current_sample[2][0, c]), ),
                      predictions, new_state)
            new_samples.append(ns)
        samples = new_samples
        display_samples(samples, encoder)
        current_depth += 1


def load_model(folder):
    with open(os.path.join(folder, 'model.pickle'), 'r') as f:
        model_dict = pickle.load(f)
    things = model(vocabulary_size=model_dict['encoder'].vocabulary_size,
                   num_unrollings=model_dict['num_unrollings'],
                   unroll_shift=model_dict['unroll_shift'],
                   batch_size=model_dict['batch_size'],
                   n_nodes=model_dict['n_nodes'])
    return things, model_dict


def lstm_layer(inputs, sampler_input, sampler_output, sampler_state, prev_n_nodes, n_nodes, batch_size, dropout_keep_prob):
    # Parameters:
    # Input gate: input, previous output, and bias.
    stddev = 0.01 / np.sqrt(n_nodes)
    forget_bias = 0.  # some say 1 helps to not stick to state at the beginning of training
    # We clump together, input, forget, update and output weights
    all_x = tf.get_variable("all_x", [prev_n_nodes, 4 * n_nodes], initializer=tf.truncated_normal_initializer(mean=0., stddev=stddev))
    all_m = tf.get_variable("all_m", [n_nodes, 4 * n_nodes], initializer=tf.truncated_normal_initializer(mean=0., stddev=stddev))
    all_b = tf.get_variable("all_b", [4 * n_nodes], initializer=tf.constant_initializer(0.))
    # Variables saving state across unrollings.
    saved_output = tf.get_variable("saved_output", [batch_size, n_nodes], initializer=tf.constant_initializer(0.), trainable=False)
    saved_state = tf.get_variable("saved_state", [batch_size, n_nodes], initializer=tf.constant_initializer(0.), trainable=False)

    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        all_gates = tf.matmul(i, all_x) + tf.matmul(o, all_m) + all_b
        input, forget, update, output = tf.split(1, 4, all_gates)
        input_gate = tf.sigmoid(input)
        forget_gate = tf.sigmoid(forget + forget_bias)
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(output)
        return output_gate * tf.tanh(state), state

    # Unrolled LSTM loop.
    outputs = list()
    states = list()
    output = saved_output
    state = saved_state
    for roll_idx, i in enumerate(inputs):
        output, state = lstm_cell(i, output, state)
        output = tf.nn.dropout(output, dropout_keep_prob, name="dropout_logits_%i" % roll_idx)
        outputs.append(output)
        states.append(state)

    sampler_output, sampler_state = lstm_cell(
        sampler_input, sampler_output, sampler_state)
    return outputs, states, saved_output, saved_state, sampler_output, sampler_state


def model(vocabulary_size, num_unrollings, unroll_shift, batch_size, n_nodes):
    assert 0 <= unroll_shift < num_unrollings
    n_layers = len(n_nodes)
    things = {}
    things['graph'] = tf.Graph()
    with things['graph'].as_default():
        # Input data.
        things['train_data'] = list()
        for j in range(num_unrollings + 1):
            things['train_data'].append(
              tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size], name='train_data_%i' % j))
        train_inputs = things['train_data'][:num_unrollings]
        train_labels = things['train_data'][1:]  # labels are inputs shifted by one time step.
        things['sampler_input'] = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='sampler_input')
        things['dropout_keep_prob'] = tf.placeholder("float", name="dropout_keep_prob")

        all_n_nodes = [vocabulary_size] + n_nodes + [vocabulary_size]
        assignments = []
        outputs = train_inputs
        sampler_input = things['sampler_input']
        sampler_combined_lengths = np.kron(n_nodes, [1, 1])  # output, state, output, state... for each sampler layer
        sampler_slices = np.hstack(([0], np.cumsum(sampler_combined_lengths)))
        things['sampler_combined_start'] = tf.zeros(shape=[1, sampler_slices[-1]], dtype=tf.float32)
        sampler_combined_list = []
        for j in range(1, n_layers + 1):
            with tf.variable_scope("lstm_%i" % j):
                sampler_output = tf.slice(things['sampler_combined_start'], [0, sampler_slices[2 * (j - 1)]], [1, sampler_combined_lengths[2 * (j - 1)]])
                sampler_state = tf.slice(things['sampler_combined_start'], [0, sampler_slices[2 * (j - 1) + 1]], [1, sampler_combined_lengths[2 * (j - 1) + 1]])
                outputs, states, saved_output, saved_state, sampler_output, sampler_state =\
                    lstm_layer(outputs, sampler_input, sampler_output, sampler_state,
                               all_n_nodes[j-1], all_n_nodes[j],
                               batch_size, things['dropout_keep_prob'])
                sampler_input = sampler_output
                sampler_combined_list += [sampler_output, sampler_state]
            assignments.append(saved_output.assign(outputs[unroll_shift]))
            assignments.append(saved_state.assign(states[unroll_shift]))

        things['sampler_combined_end'] = tf.concat(1, sampler_combined_list)
        next_stddev = 0.01 / np.sqrt(all_n_nodes[j + 1])
        with tf.variable_scope("lstm_%i" % n_layers):
            # Classifier weights and biases
            w = tf.get_variable("w", [all_n_nodes[j], all_n_nodes[j + 1]], initializer=tf.truncated_normal_initializer(mean=0., stddev=next_stddev))
            b = tf.get_variable("b", [all_n_nodes[j + 1]], initializer=tf.constant_initializer(0.))
        all_labels = tf.concat(0, train_labels)
        with tf.control_dependencies(assignments):  # we have a circular graph and this makes sure it "ends" in the loss?
            all_logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b, name="logits")
            things['loss'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(all_logits, all_labels))
        sampler_logits = tf.nn.xw_plus_b(sampler_output, w, b, name="sampler_logits")
        things['sampler_prediction'] = tf.nn.softmax(sampler_logits)

        # Optimizer.
        global_step = tf.Variable(0)
        things['learning_rate'] = tf.train.exponential_decay(
            0.01, global_step, 6000, 0.5, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(things['learning_rate'], decay=0.9)
#        optimizer = tf.train.AdamOptimizer(things['learning_rate'])
        gradients, v = zip(*optimizer.compute_gradients(things['loss'],
                                                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        things['optimizer'] = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions.
        things['train_prediction'] = tf.nn.softmax(all_logits)
        things['saver'] = tf.train.Saver()

    return things


def lstm_demo(text, model_folder=None, model_dict=None):
#    url = 'http://mattmahoney.net/dc/'
#    filename = maybe_download(url, 'text8.zip', 31344016)
#    text = np.fromstring(read_data(filename), dtype=np.uint8)
    valid_size = 5000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print('Data size %d' % len(text))
    print train_size, train_text[:64].tostring().decode('mac-roman')
    print valid_size, valid_text[:64].tostring().decode('mac-roman')

    if model_folder is None:
        t0_s=time.strftime("%c")
        model_folder = '/home/ubuntu/lstm/' + t0_s
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    instance_filename = get_latest_instance(model_folder)
    if instance_filename is None:  # nothing to continue from
        assert model_dict is not None, "Must provide model_dict for new training"
        with open(os.path.join(model_folder, 'model.pickle'), 'w') as f:
            pickle.dump(model_dict, f)
        start_iter = 0
    else:
        assert model_dict is None, ("Loading model from folder %s, cannot use given model_dict" % model_folder)
        things, model_dict = load_model(model_folder)
        start_iter = int(instance_filename[-5:]) + 1

    encoder = model_dict['encoder']
    n_nodes = model_dict['n_nodes']
    batch_size = model_dict['batch_size']
    num_unrollings = model_dict['num_unrollings']
    unroll_shift = model_dict['unroll_shift']
    dropout_keep_prob = model_dict['dropout_keep_prob']
    with open(os.path.join(model_folder, 'model.pickle'), 'w') as f:
        pickle.dump(model_dict, f)
    things = model(vocabulary_size=encoder.vocabulary_size, num_unrollings=num_unrollings,
                   batch_size=batch_size, unroll_shift=unroll_shift, n_nodes=n_nodes)
    print('Vocabulary size %d' % encoder.vocabulary_size)
    train_batches = batch_generator(train_text, batch_size, num_unrollings, encoder, unroll_shift)
    valid_batches = batch_generator(valid_text, 1, 1, encoder, 0)

    print_batches(train_batches.next(), encoder)
    print_batches(train_batches.next(), encoder)
    print_batches(valid_batches.next(), encoder)
    print_batches(valid_batches.next(), encoder)

    num_steps = 100001
    summary_frequency = 100

    with tf.Session(graph=things['graph']) as session:
        summary = tf.scalar_summary("loss", things['loss'])
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/lstm_logs", session.graph_def)
        if instance_filename is None:  # initialize from scratch
            tf.initialize_all_variables().run()
            print('Initialized')
        else:  # restore from saved
            things['saver'].restore(session, os.path.join(model_folder, instance_filename))
            print('Restored to iteration %i' % (start_iter - 1))
        mean_loss = 0
        t0 = time.time()
        for step in range(start_iter, num_steps):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[things['train_data'][i]] = batches[i]
            feed_dict[things['dropout_keep_prob']] = dropout_keep_prob
            _, l, predictions, lr, summary_str = session.run(
                [things['optimizer'], things['loss'], things['train_prediction'],
                 things['learning_rate'], merged], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                writer.add_summary(summary_str, step)
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print "Avg batch time: %.2f s" % ((time.time() - t0) / summary_frequency)
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
                if step % (summary_frequency * 10) == 0:
                    things['saver'].save(session, os.path.join(model_folder, 'instance_%.5i' % step))
                    # Generate some samples.
                    for temperature in [1, 0.001]:
                        print('=' * 30 + ('temperature=%.2f' % temperature) + '=' * 30)
                        generator = text_generator(session, things, encoder,
                                                   [np.random.randint(0, encoder.vocabulary_size)], temperature)
                        for _ in range(5):
                            sentence = ''
                            for _ in range(80):
                                sentence += generator.next()
                            print sentence
                        print('=' * 80)
                    # Measure validation set perplexity.
                    eval_feed_dict = {v: np.zeros(v._shape) for v in [things['sampler_combined_start']]}
                    eval_feed_dict[things['dropout_keep_prob']] = 1.0
                    run_list = [things['sampler_combined_end'], things['sampler_prediction']]
                    valid_logprob = 0
                    for _ in range(valid_size):
                        b = valid_batches.next()
                        eval_feed_dict[things['sampler_input']] = b[0]
                        eval_feed_dict[things['sampler_combined_start']], predictions = session.run(run_list, feed_dict = eval_feed_dict)
                        valid_logprob = valid_logprob + logprob(predictions, b[1])
                    print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))
                t0 = time.time()


def load_and_write(folder, instance_filename, start_sentence, temperature):
    things, model_dict = load_model(folder)
    print model_dict
    with tf.Session(graph=things['graph']) as session:
        things['saver'].restore(session, os.path.join(folder, instance_filename))
        seed = model_dict['encoder'].char2id(np.fromstring(start_sentence, dtype=np.uint8))
        writer = text_generator(session, things, model_dict['encoder'], seed, temperature, depth=50, n_samples=100)
        for l in range(100):
            for _ in range(80):
                sys.stdout.write(writer.next())
            print



'''
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter

'''

def get_latest_instance(folder):
    candidates = [f for f in os.listdir(folder) if f[-4:] == 'meta']
    times = [(os.stat(os.path.join(folder, c))[stat.ST_CTIME], os.path.splitext(c)[0]) for c in candidates]
    if len(times):
        sorted_times = sorted(times)
        return sorted_times[-1][1]
    else:
        return None


if __name__ == "__main__":

    # data_folder='/media/psf/Home/Downloads/asimo'
    # text = np.fromstring(compile_data(data_folder), dtype=np.uint8)
    # text[text==ord('\r')] = ord('\n')
    # num_unrollings = 60
    # model_dict = dict(encoder=Encoder(text), num_unrollings=num_unrollings,
    #                   batch_size=50, unroll_shift=num_unrollings - 1, n_nodes=[512, 512, 512],
    #                   dropout_keep_prob=1.0)
    # lstm_demo(text=text,
    #           model_folder='/home/ubuntu/lstm/512_512_512_unroll60_drop10_rmsprop',
    #           model_dict=model_dict)
    #
    #
    base_folder = '/home/ubuntu/lstm/'
    for folder in os.listdir(base_folder):
        full_folder = os.path.join(base_folder, folder)
        instance = get_latest_instance(full_folder)
        if not folder.startswith('512_512_512') or not folder.endswith('bigdata'):
            continue
        if instance is None:
            print "No valid instance in folder %s" % full_folder
            continue
        print "=" * 80
        print folder, " ", instance
        print "=" * 80
        load_and_write(full_folder, instance,
                       'Casi toda la cristiandad occidental estaba sometida a los reyes francos, y los francos no',
                       0.5)
