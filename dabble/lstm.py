import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves.urllib.request import urlretrieve


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
    def __init__(self):
        self.charlist = ['.', ',', ';', ':', '/', "'", '"', '(', ')', '\r', chr(151), ]
        self.vocabulary_size = len(self.charlist) + 1 + ord('z') - ord('a') + 1

    def char2id(self, chars):
        ids = np.zeros_like(chars)
        for i, c in enumerate(self.charlist):
            ids[chars == ord(c)] = i + 1
        alphanum = (chars >= ord('a')) & (chars <= ord('z'))
        ids[alphanum] = chars[alphanum] - ord('a') + len(self.charlist) + 1
        return ids

    def id2char(self, single_id):
        if single_id == 0:
            return ' '
        elif single_id <= len(self.charlist):
            return self.charlist[single_id - 1]
        else:
            return chr(single_id - len(self.charlist) - 1 + ord('a'))


def batch_generator(text, batch_size, num_unrollings, encoder, unroll_shift):
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
  return [encoder.id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches, encoder):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b, encoder))]
  return s


def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1


def sample(prediction, vocabulary_size):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p


def random_distribution(vocabulary_size):
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


def model(vocabulary_size, num_unrollings, unroll_shift, batch_size):
    assert 0 <= unroll_shift < num_unrollings
    num_nodes = 128
    things = {}
    things['graph'] = tf.Graph()
    with things['graph'].as_default():

        # Parameters:
        # Input gate: input, previous output, and bias.
        ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output, and bias.
        fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, state and bias.
        cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias.
        ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))

        def lstm_cell(i, o, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state


        # Input data.
        things['train_data'] = list()
        for _ in range(num_unrollings + 1):
            things['train_data'].append(
              tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
        train_inputs = things['train_data'][:num_unrollings]
        train_labels = things['train_data'][1:]  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        states = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)
            states.append(state)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(outputs[unroll_shift]),
                                      saved_state.assign(states[unroll_shift])]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            things['loss'] = tf.reduce_mean(
              tf.nn.softmax_cross_entropy_with_logits(
                logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)
        things['learning_rate'] = tf.train.exponential_decay(
            5.0, global_step, 20000, 0.5, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(things['learning_rate'])
        gradients, v = zip(*optimizer.compute_gradients(things['loss']))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        things['optimizer'] = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions.
        things['train_prediction'] = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        things['sample_input'] = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        things['reset_sample_state'] = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(
            things['sample_input'], saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            things['sample_prediction'] = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
    return things


def lstm_demo():
#    url = 'http://mattmahoney.net/dc/'
#    filename = maybe_download(url, 'text8.zip', 31344016)
#    text = np.fromstring(read_data(filename), dtype=np.uint8)
    folder = '/media/psf/Home/Downloads/asimo'
    text = np.fromstring(compile_data(folder), dtype=np.uint8)
    print('Data size %d' % len(text))
    valid_size = 5000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print(train_size, train_text[:64].tostring())
    print(valid_size, valid_text[:64].tostring())
    encoder = Encoder()
    batch_size=60
    num_unrollings=50
    unroll_shift = num_unrollings - 1

    train_batches = batch_generator(train_text, batch_size, num_unrollings, encoder, unroll_shift)
    valid_batches = batch_generator(valid_text, 1, 1, encoder, 0)

    print(batches2string(train_batches.next(), encoder))
    print(batches2string(train_batches.next(), encoder))
    print(batches2string(valid_batches.next(), encoder))
    print(batches2string(valid_batches.next(), encoder))

    num_steps = 100001
    summary_frequency = 1000
    things = model(vocabulary_size=encoder.vocabulary_size, num_unrollings=num_unrollings,
                   batch_size=batch_size, unroll_shift=unroll_shift)

    with tf.Session(graph=things['graph']) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[things['train_data'][i]] = batches[i]
            _, l, predictions, lr = session.run(
                [things['optimizer'], things['loss'], things['train_prediction'], things['learning_rate']], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution(encoder.vocabulary_size), encoder.vocabulary_size)
                        sentence = characters(feed, encoder)[0]
                        things['reset_sample_state'].run()
                        for _ in range(79):
                            prediction = things['sample_prediction'].eval({things['sample_input']: feed})
                            feed = sample(prediction, vocabulary_size=encoder.vocabulary_size)
                            sentence += characters(feed, encoder)[0]
                        print(sentence)
                    print('=' * 80)
                    # Measure validation set perplexity.
                    things['reset_sample_state'].run()
                    valid_logprob = 0
                    for _ in range(valid_size):
                        b = valid_batches.next()
                        predictions = things['sample_prediction'].eval({things['sample_input']: b[0]})
                        valid_logprob = valid_logprob + logprob(predictions, b[1])
                    print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


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

if __name__ == "__main__":
    lstm_demo()