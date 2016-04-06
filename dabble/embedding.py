import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


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
    data = tf.compat.as_str(f.read(name)).split()
    f.close()
    return data


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = np.empty(len(words), dtype=np.int)
    for i, word in enumerate(words):
        data[i] = dictionary[word] if word in dictionary else 0  # 0 being 'UNK'
    unk_count = np.count_nonzero(data==0)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def word2vec_batch_generator(data, batch_size, num_skips, skip_window):
    data_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    while True:
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        yield batch, labels


def cbow_batch_generator(data, batch_size, skip_window):
    data_index = 0
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    base = np.linspace(0, span - 1e-7, span - 1).astype(np.int)[None, :] + np.arange(batch_size, dtype=np.int)[:, None]
    while True:
        batch = data[(base + data_index) % len(data)]
        labels = data[(np.arange(batch_size, dtype=np.int) + data_index + skip_window) % len(data)][:, None]
        data_index += 1
        yield batch, labels


def plot_embeddings(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()


def model(vocabulary_size, format='word2vec'):
    assert format in ['word2vec', 'cbow']
    embedding_size = 128 # Dimension of the embedding vector.
    num_sampled = 64 # Number of negative examples to sample.
    gradient_speed = {'word2vec': 1., 'cbow': 0.1}[format]
    things = {}
    things['graph'] = tf.Graph()
    with things['graph'].as_default():
        # Input data.
        things['input'] = tf.placeholder(tf.int32)
        things['output'] = tf.placeholder(tf.int32)

        # Variables.
        things['embeddings'] = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                 stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Model.
        # Look up embeddings for inputs.
        embedded_input = tf.nn.embedding_lookup(things['embeddings'], things['input'])
        if format == 'cbow':
            embedded_input = tf.reduce_mean(embedded_input, reduction_indices=[1])
        # Compute the softmax loss, using a sample of the negative labels each time.
        things['loss'] = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embedded_input,
                                       things['output'], num_sampled, vocabulary_size))

        things['optimizer'] = tf.train.AdagradOptimizer(gradient_speed).minimize(things['loss'])

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(things['embeddings']), 1, keep_dims=True))
        things['normalized_embeddings'] = things['embeddings'] / norm
        normalized_embedded_input = tf.nn.embedding_lookup(things['normalized_embeddings'], things['input'])
        things['similarity'] = tf.matmul(normalized_embedded_input, tf.transpose(things['normalized_embeddings']))
    return things


def demo(format):
    assert format in ['word2vec', 'cbow']
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download(url, 'text8.zip', 31344016)
    vocabulary_size = 50000
    words = read_data(filename)
    print('Data size %d' % len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.filename = maybe_download('text8.zip', 31344016)
    print('data:', [reverse_dictionary[di] for di in data[:8]])

    batch_size = 128
    skip_window = 1 # How many words to consider left and right.
    num_skips = 2 # How many times to reuse an input to generate a label (only word2vec)
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100 # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    things = model(vocabulary_size, format=format)

    num_steps = {'word2vec': 100001, 'cbow': 1000001}[format]
    batch_generator = {'word2vec': word2vec_batch_generator(data, batch_size, num_skips, skip_window),
                       'cbow': cbow_batch_generator(data, batch_size, skip_window)}[format]
    with tf.Session(graph=things['graph']) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = batch_generator.next()
            feed_dict = {things['input'] : batch_data, things['output'] : batch_labels}
            _, l = session.run([things['optimizer'], things['loss']], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                feed_dict = {things['input'] : valid_examples}
                sim = session.run(things['similarity'], feed_dict=feed_dict)
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = things['normalized_embeddings'].eval()

    num_points = 400
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
    plot_embeddings(two_d_embeddings, words)


if __name__ == "__main__":
    demo('cbow')