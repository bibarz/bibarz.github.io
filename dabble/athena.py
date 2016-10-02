import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
import scipy.weave
import matplotlib.pyplot as plt
import cPickle
import os
import cv2
import time
import gc
from collections import defaultdict
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier


def monto_vec(x, idx):
    return x.groupby('DAY').sum()['MONTO'].reindex(idx).fillna(0)


def tipo_vec(x, idx):
    return x.groupby('DAY').first()['TIPO_IDX'].reindex(idx).fillna(255)


def conv_relu(input, kernel_shape, stride):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.truncated_normal_initializer(
        mean=0., stddev=0.01 / np.sqrt(np.prod(kernel_shape[:3]))))
    biases = tf.get_variable("biases", kernel_shape[-1:], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, [1, stride, 1], padding='SAME')
#   conv_max = tf.nn.avg_pool(conv, [1, stride, 1], [1, stride, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def model(data, data_length, n_channels, n_outputs):
    kernel_defs = [(8, 8, 2), (2, 8, 2)]  # each conv layer, (patch_side, n_kernels, stride)
    fc_sizes = [128]
    n_input_kernels = n_channels
    for i, k in enumerate(kernel_defs):
        with tf.variable_scope("conv_%i" % i):
            kernel_shape = (k[0], n_input_kernels, k[1])
            data = conv_relu(data, kernel_shape, k[2])
            n_input_kernels = k[1]

    for i, n in enumerate(fc_sizes):
        with tf.variable_scope("fc_%i" % i):
            if i == 0:
                previous_n = kernel_defs[-1][1] * data_length / np.prod([k[2] for k in kernel_defs])**2
                data = tf.reshape(data, [-1, previous_n])
            else:
                previous_n = fc_sizes[i-1]
            weights = tf.get_variable("weights", [previous_n, n],
                                      initializer=tf.truncated_normal_initializer(mean=0., stddev=0.01 / np.sqrt(previous_n)))
            biases = tf.get_variable("biases", [n], initializer=tf.constant_initializer(0.0))
            relu_input = tf.matmul(data, weights) + biases
            data = tf.nn.relu(relu_input)

    with tf.variable_scope("flat_out"):
        weights = tf.get_variable("weights", [fc_sizes[-1], n_outputs],
                                  initializer=tf.truncated_normal_initializer(mean=0., stddev=0.01 / np.sqrt(fc_sizes[-1])))
        biases = tf.get_variable("biases", [n_outputs], initializer=tf.constant_initializer(0.0))
        return tf.matmul(data, weights) + biases


def make_learner(data_length, n_channels, n_outputs):
    things = {}
    things['graph'] = tf.Graph()
    with things['graph'].as_default():

        # Input and teacher place holders
        things['input'] = tf.placeholder(tf.float32)
        things['output'] = tf.placeholder(tf.float32)

        things['logits'] = model(things['input'], data_length, n_channels, n_outputs)
        things['loss'] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(things['logits'], things['output']))

        things['learner'] = tf.train.AdamOptimizer(0.001).minimize(things['loss'])

        # Predictors for the training, validation, and test data.
        things['prediction'] = tf.nn.softmax(things['logits'])
        things['saver'] = tf.train.Saver()

    return things


class Convnet(object):
    def __init__(self, data_length, n_channels, n_outputs, batch_size, n_passes):
        self.things = make_learner(data_length, n_channels, n_outputs)
        self._batch_size = batch_size
        self._n_passes = n_passes

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_training = len(X)
        n_steps = (n_training * self._n_passes) / self._batch_size
        things = self.things
        with tf.Session(graph=things['graph']) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            train_idx = np.random.permutation(n_training)
            for step in range(n_steps):
                offset = (step * batch_size) % n_training
                if n_training - offset > self._batch_size:
                    idx = np.concatenate((train_idx[offset:], training_idx[:self._batch_size - (n_training - offset)]))
                else:
                    idx = train_idx[offset:(offset + self._batch_size)]
                batch_data = X[idx]
                batch_labels = y[idx]
                feed_dict = {things['input']: batch_data, things['output']: batch_labels}
                session.run(things['learner'], feed_dict=feed_dict)
                if (step % (n_steps / 10) == 0):
                    l, predictions = session.run(
                        [things['loss'], things['prediction']], feed_dict=feed_dict)
                    print('Minibatch loss at step %d/%d: %.3f, accuracy %.1f%%' % (step, n_steps, l,
                        100 * np.mean((predictions > 0.5) == batch_labels)))

    def predict(self, X):
        X = check_array(X)
        things = self.things
        feed_dict = {things['input']: X}
        with tf.Session(graph=things['graph']) as session:
            l, predictions = session.run(
                [things['loss'], things['prediction']], feed_dict=feed_dict)
        return (l > 0.5)


def save_monto_matrix(filename):
    '''
    Read data, save in pickle as n_users x n_days DataFrame of montos
    '''
    nrows = None
    t0 = time.time()
    a=pd.read_csv('/media/psf/Home/linux-home/Borja/Cursos/kaggle/athena/dataset.recruiting.20160816.ts.csv',
                  sep='|', header=0, parse_dates=[0, 5], dtype={'MONTO': np.float32}, nrows=nrows)
    print "Read took", time.time()-t0
    time_idx = pd.DatetimeIndex(start=pd.datetime(2014, 1, 1), end=pd.datetime(2014, 5, 31), freq='D')
    g = a.groupby('ABONADO')
    t0 = time.time()
    v = g.apply(lambda x: monto_vec(x, time_idx))
    print "Matrix took", time.time()-t0
    with open(filename, 'w') as f:
        cPickle.dump(v, f, protocol=-1)


def save_tipo_matrix(filename):
    '''
    Read data, save in pickle as n_users x n_days DataFrame of tipo de recarga
    '''
    nrows = None
    t0 = time.time()
    a=pd.read_csv('/media/psf/Home/linux-home/Borja/Cursos/kaggle/athena/dataset.recruiting.20160816.ts.csv',
                  sep='|', header=0, parse_dates=[0, 5], dtype={'MONTO': np.float32}, nrows=nrows)
    print "Read took", time.time()-t0
    time_idx = pd.DatetimeIndex(start=pd.datetime(2014, 1, 1), end=pd.datetime(2014, 5, 31), freq='D')
    unique_tipos = pd.Series(a.DES_TIPO_RECARGA.unique())
    tipo_idx = pd.Series(data=unique_tipos.index, index=unique_tipos.values)
    a['TIPO_IDX'] = a.DES_TIPO_RECARGA.map(tipo_idx)
    b = a[['ABONADO', 'DAY', 'TIPO_IDX']]
    g = b.groupby('ABONADO')
    t0 = time.time()
    v = g.apply(lambda x: tipo_vec(x, time_idx)).astype(np.uint8)
    print "Matrix took", time.time()-t0

    with open(filename, 'w') as f:
        cPickle.dump((unique_tipos, v), f, protocol=-1)


def common_tipo(v, n_values=2):
    '''
    For each row return most common value ignoring 255
    n_values: number of possible types other than 255
    '''
    medians = np.empty(len(v), dtype=v.dtype)
    code = '''
        int counts[n_values];
        for (int i = 0; i < Nv[0]; ++i) {
            memset((void*)counts, 0, n_values * sizeof(int));
            for (int j=0; j < Nv[1]; ++j) {
                if (v(i, j) != 255) {
                    counts[v(i, j)] += 1;
                }
            }
            int max = 0;
            int imax = 255;
            for (int j=0; j < n_values; ++j) {
                if (counts[j] > max) {
                    max = counts[j];
                    imax = j;
                }
            }
            medians(i) = imax;
        }
    '''
    scipy.weave.inline(
        code, ['v', 'n_values', 'medians'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return medians


def cumhist_montos(values):
    minmax = np.amin(values), np.amax(values)
    range = minmax[1] - minmax[0]
    nbins = 30000
    h, edges = np.histogram(values.ravel(), bins=nbins, density=True, range=[minmax[0] - 0.25 * range/nbins,
                                                                             minmax[1] + 0.25 * range/nbins])
    cumhist = np.concatenate(([0], np.cumsum(h) * np.mean(np.diff(edges))))
    return cumhist, edges


def bin_montos(values, n_classes):
    cumhist, edges = cumhist_montos(values)
    levels = np.linspace(0, 1, n_classes + 1)
    idx = np.unique(np.argmin(np.abs(levels[:, None] - cumhist[None, :]), axis=1))
    assert idx[0] == 0
    assert idx[-1] == len(cumhist) - 1
    bins = edges[idx] + 0.125 * range / n_classes
    # h2, edges2 = np.histogram(values.ravel(), bins=bins)
    # print "distribution:", h2
    return bins


def group_daily(monto, tipo, churn):
    return monto.values, tipo.values, churn


def group_weekly(monto, tipo, churn):
    dates = monto.T.index
    newdates = dates - np.array(dates.day - np.clip((dates.day - 1) // 7, 0, 3) * 7 - 1, dtype="timedelta64[D]")
    _, idx = np.unique(newdates.values, return_index=True)
    idx = np.concatenate((idx, [len(dates)]))
    agg_montos = np.empty((len(monto), len(idx) - 1))
    agg_tipos = np.empty((len(monto), len(idx) - 1))
    for i in range(len(idx) - 1):
        agg_montos[:, i] = monto.values[:, idx[i]:idx[i + 1]].sum(axis=1)
        agg_tipos[:, i] = common_tipo(tipo.values[:, idx[i]:idx[i + 1]])
    return agg_montos, agg_tipos, churn


def group_rbf(monto, tipo, churn, period):
    x = np.arange(-3 * period, 3 * period + 1).astype(np.float32)
    rbf = np.exp(-x ** 2 / 2 / period **2)
    rbf /= np.sum(rbf)
    agg_montos = cv2.filter2D(monto.values, cv2.CV_32F, rbf[None, :], borderType=cv2.BORDER_CONSTANT)
    agg_montos = agg_montos[:, period::(period + period / 2)]
    agg_tipos = common_tipo(tipo.values)
    return agg_montos, agg_tipos, churn


def group_fft(monto, tipo, churn):
    fft = np.fft.rfft(monto.values, axis=1)
    n_coeff = monto.shape[1] / 10  # reject high frequency
    fft_montos = np.hstack((np.real(fft[:, :n_coeff]), np.imag(fft)[:, 1:n_coeff]))
    agg_tipos = common_tipo(tipo.values)
    return fft_montos, agg_tipos, churn


def discretize(monto, tipo, churn, n_classes, with_tipo=False):
    bins = bin_montos(monto, n_classes=n_classes)
    # print "n_bins: %i, bins: [%s]" % (len(bins) - 1, ",".join(["%.2f" % s for s in bins]))
    discretized = np.digitize(monto, bins).astype(np.uint8)
    if with_tipo:
        tot_0 = np.sum(tipo==0, axis=1)
        tot_1 = np.sum(tipo==1, axis=1)
        zerrer = tot_0 > tot_1
        onerr = tot_1 > tot_0
        print "0ers:", np.sum(zerrer), "churn 0:", 100 * np.mean(churn[zerrer])
        print "1ers:", np.sum(onerr), "churn 1:", 100 * np.mean(churn[onerr])
        discretized = np.hstack((discretized, tipo[:, None]))
    return discretized, churn


def discretize_columnwise(monto, tipo, churn, n_classes, with_tipo=False):
    discretized = np.empty(monto.shape, dtype=np.uint8)
    for k in range(monto.shape[1]):
        bins = bin_montos(monto[:, k], n_classes=n_classes)
        print "k=%i, n_bins: %i, bins: [%s]" % (k, len(bins) - 1, ",".join(["%.2f" % s for s in bins]))
        discretized[:, k] = np.digitize(monto[:, k], bins)
    if with_tipo:
        tot_0 = np.sum(tipo==0, axis=1)
        tot_1 = np.sum(tipo==1, axis=1)
        zerrer = tot_0 > tot_1
        onerr = tot_1 > tot_0
        print "0ers:", np.sum(zerrer), "churn 0:", 100 * np.mean(churn[zerrer])
        print "1ers:", np.sum(onerr), "churn 1:", 100 * np.mean(churn[onerr])
        discretized = np.hstack((discretized, tipo[:, None]))
    return discretized, churn


def equalize(monto, tipo, churn, with_tipo=False):
    '''
    Equalize monto distribution between -1 and 1
    '''
    cumhist, edges = cumhist_montos(monto)
    equalized = np.interp(monto, edges, cumhist)
    equalized = (equalized - np.mean(equalized)) / (np.std(equalized) + 1e-6)
    if with_tipo:
        tipo_center = np.zeros(tipo.shape, dtype=np.float32)
        tipo_center[tipo == 0] = -1
        tipo_center[tipo == 1] = 1
        tipo_center = (tipo_center - np.mean(tipo_center)) / (np.std(tipo_center) + 1e-6)
        tot_0 = np.sum(tipo==0, axis=1)
        tot_1 = np.sum(tipo==1, axis=1)
        zerrer = tot_0 > tot_1
        onerr = tot_1 > tot_0
        print "0ers:", np.sum(zerrer), "churn 0:", 100 * np.mean(churn[zerrer])
        print "1ers:", np.sum(onerr), "churn 1:", 100 * np.mean(churn[onerr])
        equalized = np.hstack((equalized, tipo_center[:, None]))
    return equalized, churn


def predict_churn(n_months_to_use, grouper, conditioner, clf_factory):
    '''
    grouper: function taking monto and tipo dataframes, and churn array,
        and returning grouped monto array, grouped tipo array, and churn
    conditioner: function taking grouped monto and tipo arrays and churn array,
        and returning discretized, equalized, scaled, centered training_data and
        labels arrays
    clf_factory: function taking number of features and returning a classifier
    '''
    with open('monto_panda.pkl', 'r') as f:
        monto = cPickle.load(f)
    with open('tipo_panda.pkl', 'r') as f:
        t_index, tipo = cPickle.load(f)
    tipo[(tipo > 0) & (tipo < 255)] = 1

    for month in range(5 - n_months_to_use):
        last_month_cols = monto.columns[monto.columns.month == month + n_months_to_use]
        training_cols = monto.columns[(monto.columns.month >= month + 1) & (monto.columns.month <= month + n_months_to_use)]
        result_cols = monto.columns[monto.columns.month == month + 1 + n_months_to_use]
        good = (monto.loc[:, last_month_cols] != 0).any(axis=1)
        training_monto = monto.loc[good, training_cols]
        training_tipo = tipo.loc[good, training_cols]
        result_monto = monto.loc[good, result_cols]
        churn = (result_monto == 0).all(axis=1)
        print "month %i, good are %i of %i; churners are %i of %i" % (month, good.sum(), len(good), churn.sum(), len(churn))

        grouped_monto, grouped_tipo, churn = grouper(training_monto, training_tipo, churn)
        training_data, labels = conditioner(grouped_monto, grouped_tipo, churn)

        n_feat = training_data.shape[1]
        clf = clf_factory(n_feat)
        score = cross_val_score(clf, training_data, churn, cv=3, scoring='roc_auc')
        print "score:", score


def tipo_histogram():
    with open('tipo_panda.pkl', 'r') as f:
        t_index, v = cPickle.load(f)
    vv = v.values.copy()
    vv[(vv > 0) & (vv < 255)] = 1
    autos = np.sum(vv == 0, axis=1)
    others = np.sum(vv == 1, axis=1)
    percent = 100 * autos.astype(np.float) / (autos + others)
    pd.Series(percent).hist(bins=100)
    plt.xlabel('% insercion automatica')
    plt.ylabel('number of users')
    plt.show()


if __name__ == "__main__":
    # bin_montos('7D')
    forest = lambda n_feat: RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                   max_features=min(n_feat, max(4, 2 * np.ceil(np.sqrt(n_feat)).astype(np.int))),
                                                   max_depth=10)
    dummy = lambda n_feat: DummyClassifier()
    convnet = lambda n_feat: Convnet(n_feat, 1, 1, batch_size=256, n_passes=2)

    # predict_churn(n_months_to_use=1, groupr=group_weekly)
    predict_churn(n_months_to_use=3, grouper=group_fft,
                  conditioner=lambda a, b, c: discretize_columnwise(a, b, c, n_classes=10, with_tipo=False),
                  clf_factory=forest)
    # predict_churn(n_months_to_use=1, groupr=lambda a, b, c: group_rbf(a, b, c, 7))
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # try:
    #     profiler.runctx('main()', globals(), locals())
    # finally:
    #     profiler.dump_stats('prof')
    #
    # p = pstats.Stats("prof")
    # p.strip_dirs().sort_stats('cumtime').print_stats(100)
    # # # p.print_callers('isinstance')
