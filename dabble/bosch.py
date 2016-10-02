import numpy as np
import pandas as pd
import itertools
import scipy.weave
import cPickle
import os
import time
import gc
from collections import defaultdict
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier


def pca_converter(data, feature_discriminabilities, explained_variance):
    '''
    PCA conversion of the data. The PCA is based on the complete dataset, but each feature
        is normalized to a std dev proportional to the given discriminability.
    :param data: n_samples x n_features matrix with all data to do PCA on
    :param feature_discriminabilities: n_features length vector
    :param explained_variance: ratio of explained variance (between 0 and 1) that will
        determine how many components are kept
    :return: function transforming data into pca components, and covariance matrix
        of transformed data
    '''
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0) / feature_discriminabilities
    normalized_data = (data - mu) / (1e-9 + std)
    u, s, vt = np.linalg.svd(normalized_data)
    cut_idx = np.argmin(np.abs(np.cumsum(s * s) / np.sum(s * s) - explained_variance))
    vt = vt[:cut_idx + 1]
    return (lambda x, mu=mu, std=std, vt=vt: np.dot((x - mu) / (1e-9 + std), vt.T)),\
           np.diag(s[:cut_idx + 1] ** 2 / (len(data) - 1))


def unite(g):
    return set.union(*[set(s) for s in g])


def min_max_jaccard(frame):
    if len(frame) == 1:
        return pd.Series(dict(set_count=1, min_j=1., max_j=1., mean_j=1.))
    all_lengths = np.array([len(s) for s in frame['unique_feature_set']])
    set_limits = np.cumsum(np.concatenate(([0], all_lengths))).astype(np.int32)
    sets_vector = np.ascontiguousarray(np.concatenate(frame['unique_feature_set'].tolist())).astype(np.int32)
    J_minmaxmean = np.array([1., 0., 0.])
    code = '''
        for (int i = 1; i < Nall_lengths[0]; ++i) {
            int length_i = all_lengths(i);
            for (int j = 0; j < i; ++j) {
                int length_j = all_lengths(j);
                int u = length_i + length_j;
                int* p = &sets_vector(set_limits(i));
                int* end_p = &sets_vector(set_limits(i + 1));
                int* q = &sets_vector(set_limits(j));
                int* end_q = &sets_vector(set_limits(j + 1));
                double intersect = 0;
                while (p < end_p && q < end_q) {
                    if (*p < *q) ++p;
                    else if (*p > *q) ++q;
                    else {
                        intersect += 1.0;
                        ++p;
                        ++q;
                    }
                }
                double J = intersect / (u - intersect);
                if (J < J_minmaxmean(0)) J_minmaxmean(0) = J;
                if (J > J_minmaxmean(1)) J_minmaxmean(1) = J;
                J_minmaxmean(2) += J;
            }
        }
        J_minmaxmean(2) /= (Nall_lengths[0] * (Nall_lengths[0] - 1)) / 2;
    '''
    scipy.weave.inline(
        code, ['all_lengths', 'set_limits', 'sets_vector', 'J_minmaxmean'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return pd.Series(dict(set_count=len(frame), max_j=J_minmaxmean[1], min_j=J_minmaxmean[0], mean_j=J_minmaxmean[2]))


def simple_jaccard_graph(unique_feature_sets, min_jaccard):
    '''
    Cannot be used for big sets of sets (i.e., in the thousands)
    :param unique_feature_sets: list of tuples with indices of features in each set
    :param all_lengths: np.sum(m, axis=1)
    :return: list of edges connecting sets with Jaccard similarity > min_jaccard
    '''
    all_lengths = np.array([len(s) for s in unique_feature_sets])
    all_last = np.array([(s[-1] if len(s) else -1) for s in unique_feature_sets])
    m = np.zeros((len(unique_feature_sets), np.amax(all_last) + 1), dtype=np.int)  # int to allow dot product without overflow
    for i, s in enumerate(unique_feature_sets):
        m[i, s] = 1
    intersect = np.dot(m, m.T)
    union = all_lengths[None, :]  + all_lengths[:, None]  - intersect
    J = intersect.astype(np.float) / union
    ei, ej = np.where(J > min_jaccard)
    good = ei > ej
    ei = ei[good]
    ej = ej[good]
    edges = np.zeros(2 * len(ei), dtype=np.int)
    edges[::2] = ej
    edges[1::2] = ei
    return edges


def jaccard_graph(unique_feature_sets, min_jaccard):
    '''
    :param unique_feature_sets: list of tuples with indices of features in each set
    :return: list of edges connecting sets with Jaccard similarity > min_jaccard
    '''
    all_lengths = np.array([len(s) for s in unique_feature_sets])
    set_limits = np.cumsum(np.concatenate(([0], all_lengths))).astype(np.int32)
    sets_vector = np.ascontiguousarray(np.concatenate(unique_feature_sets)).astype(np.int32)
    jaccard_coeff = min_jaccard / (1 + min_jaccard)
    code = '''
        py::list edges;
        py::list jaccards;
        for (int i = 1; i < Nall_lengths[0]; ++i) {
            int length_i = all_lengths(i);
            for (int j = 0; j < i; ++j) {
                int length_j = all_lengths(j);
                int u = length_i + length_j;
                int min_intersect = (int)(u * jaccard_coeff) + 1;
                if (std::min(length_i, length_j) < min_intersect) continue;
                int* p = &sets_vector(set_limits(i));
                int* end_p = &sets_vector(set_limits(i + 1));
                int* q = &sets_vector(set_limits(j));
                int* end_q = &sets_vector(set_limits(j + 1));
                int remaining_intersect = min_intersect;
                int actual_intersect = min_intersect - 1;
                while (p <= end_p - remaining_intersect && q <= end_q - remaining_intersect) {
                    if (*p < *q) ++p;
                    else if (*p > *q) ++q;
                    else {
                        if (remaining_intersect == 1) ++actual_intersect;
                        else --remaining_intersect;
                        ++p;
                        ++q;
                    }
                }
                if (actual_intersect >= min_intersect) {
                    edges.append(j);
                    edges.append(i);
                    jaccards.append((double)(actual_intersect) / (u - actual_intersect));
                }
            }
        }
        py::tuple results(2);
        results[0] = edges;
        results[1] = jaccards;
        return_val = results;
    '''
    edges, jaccards = scipy.weave.inline(
        code, ['all_lengths', 'set_limits', 'sets_vector', 'jaccard_coeff'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return np.array(edges, dtype=np.int), np.array(jaccards)


def simple_connected_components(all_sets, edges):
    adj = np.zeros((len(all_sets), len(all_sets)), dtype=np.int)
    adj[edges[::2], edges[1::2]] = 1
    adj[edges[1::2], edges[::2]] = 1
    L = np.diag(np.sum(adj, axis=1)) - adj
    u, s, v = np.linalg.svd(L)
    null_dim = np.sum(s <= 1e-9)
    return null_dim


def connected_components(n_vertices, edges, similarity=None, n_cc=0):
    '''
    :param n_vertices:
    :param edges:
    :param similarity: if given, length edges/2, opposite of weight for an edge, higher
        similarity edges will be chosen first
    :param n_cc: clustering stops when this number of components are left (or when
        we run out of edges)
    :return:
    '''
    parent = np.arange(n_vertices, dtype=np.int)
    rank = np.zeros(n_vertices, dtype=np.int)
    if similarity is None:
        order = np.arange(len(edges) / 2)
    else:
        order = np.argsort(1 - similarity)
    code = '''
        int n_components = n_vertices;
        for (int i=0; i < Norder[0]; ++i) {
            int u = edges(order(i) * 2);
            int v = edges(order(i) * 2 + 1);
            int root_u = u;
            int root_v = v;
            while (parent(root_u) != root_u) root_u = parent(root_u);
            while (parent(root_v) != root_v) root_v = parent(root_v);
            if (root_u == root_v) continue;
            if (n_components <= n_cc) break;
            n_components -= 1;
            int rank_u = rank(root_u);
            int rank_v = rank(root_v);
            if (rank_u > rank_v) {
                parent(root_v) = root_u;
            }
            else {
                parent(root_u) = root_v;
                rank(root_v) = std::max(rank_v, 1 + rank_u);
            }
        }
        for (int i=0; i < parent.size(); ++i) {
            int root_u = parent(i);
            while (parent(root_u) != root_u) root_u = parent(root_u);
            parent(i) = root_u;
        }
    '''
    scipy.weave.inline(
        code, ['edges', 'parent', 'rank', 'order', 'n_cc', 'n_vertices'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return parent


def jaccard_dbscan(unique_feature_sets, n_clusters):
    '''
    Cluster via DBScan (not exactly: instead of a fixed max edge distance/min similarity,
        we do Prim's minimum spanning tree until we have only n_clusters left)
        based on jaccard distance
    '''
    edges, jaccards = jaccard_graph(unique_feature_sets, 0.7)  # 0.7 is low enough to make all one big cluster, or almost
    print len(edges), np.amin(jaccards), np.amax(jaccards)
    assert np.all(jaccards >= 0.7)
    # edges_2 = simple_jaccard_graph(unique_feature_sets, 0.9)
    # assert np.array_equal(edges, edges_2)
    cc = connected_components(len(unique_feature_sets), edges, similarity=jaccards, n_cc=n_clusters)
    # num_components = len(np.unique(cc))
    # num_components_2 = simple_connected_components(all_sets, edges)
    # assert num_components_2 == num_components
    return cc


def jaccard_kmeans(unique_feature_sets, n_clusters):
    '''
    :param unique_feature_sets: list of tuples with indices of features in each set
    :return: list of edges connecting sets with Jaccard similarity > min_jaccard
    '''
    all_lengths = np.array([len(s) for s in unique_feature_sets])
    set_limits = np.cumsum(np.concatenate(([0], all_lengths))).astype(np.int32)
    sets_vector = np.ascontiguousarray(np.concatenate(unique_feature_sets)).astype(np.int32)
    cluster_seed = np.random.permutation(len(unique_feature_sets))[:n_clusters]
    code = '''
        std::vector<int> clusters[Ncluster_seed[0]];
        int maxiter = 10;
        //printf("Jaccard Kmeans round ");
        for (int iter = 0; iter <= maxiter; ++iter) {
            //printf("%i/%i, ", iter, maxiter);
            //printf("round %i cluster seeds: ", iter);
            //for (int i = 0; i < Ncluster_seed[0]; ++i) printf("%i,", cluster_seed(i));
            //printf("\\n");

            // Phase 1: assign points to clusters
            for (int i = 0; i < Ncluster_seed[0]; ++i) clusters[i].clear();
            for (int i = 0; i < Nall_lengths[0]; ++i) {
                double max_jaccard = 0.0;
                int max_idx = 0;
                int length_i = all_lengths(i);
                int* begin_p = &sets_vector(set_limits(i));
                int* end_p = &sets_vector(set_limits(i + 1));
                for (int j = 0; j < Ncluster_seed[0]; ++j) {
                    int centroid_idx = cluster_seed(j);
                    if (centroid_idx == i) {
                        max_idx = j;
                        break;
                    }
                    int length_j = all_lengths(centroid_idx);
                    int u = length_i + length_j;
                    int min_intersect = (int)(u * max_jaccard / (1 + max_jaccard)) + 1;
                    if (std::min(length_i, length_j) < min_intersect) continue;
                    int* p = begin_p;
                    int* q = &sets_vector(set_limits(centroid_idx));
                    int* end_q = &sets_vector(set_limits(centroid_idx + 1));
                    int remaining_intersect = min_intersect;
                    int actual_intersect = min_intersect - 1;
                    while (p <= end_p - remaining_intersect && q <= end_q - remaining_intersect) {
                        if (*p < *q) ++p;
                        else if (*p > *q) ++q;
                        else {
                            if (remaining_intersect == 1) ++actual_intersect;
                            else --remaining_intersect;
                            ++p;
                            ++q;
                        }
                    }
                    if (actual_intersect >= min_intersect) {
                        max_jaccard = (double)(actual_intersect) / (u - actual_intersect);
                        max_idx = j;
                    }
                }
                clusters[max_idx].push_back(i);
            }

            //printf("round %i cluster population: ", iter);
            //for (int i = 0; i < Ncluster_seed[0]; ++i) printf("%i,", clusters[i].size());
            //printf("\\n");
            if (iter == maxiter) break;

            // Phase 2: compute new centroids
            bool has_changed = false;
            for (int i = 0; i < Ncluster_seed[0]; ++i) {
                std::vector<double> total_J(clusters[i].size(), 0);
                int centroid_idx = 0;
                double best_total_J = 0.;
                for (int j = 0; j < clusters[i].size(); ++j) {
                    int j_idx = clusters[i][j];
                    int length_j = all_lengths(j_idx);
                    int* begin_p = &sets_vector(set_limits(j_idx));
                    int* end_p = &sets_vector(set_limits(j_idx + 1));
                    for (int k = j + 1; k < clusters[i].size(); ++k) {
                        int k_idx = clusters[i][k];
                        int length_k = all_lengths(k_idx);
                        int u = length_j + length_k;
                        int* p = begin_p;
                        int* q = &sets_vector(set_limits(k_idx));
                        int* end_q = &sets_vector(set_limits(k_idx + 1));
                        int actual_intersect = 0;
                        while (p < end_p && q < end_q) {
                            if (*p < *q) ++p;
                            else if (*p > *q) ++q;
                            else {
                                ++actual_intersect;
                                ++p;
                                ++q;
                            }
                        }
                        double jaccard = (double)(actual_intersect) / (u - actual_intersect);
                        total_J[j] += jaccard;
                        total_J[k] += jaccard;
                        if (total_J[j] > best_total_J) {
                            best_total_J = total_J[j];
                            centroid_idx = j_idx;
                        }
                        if (total_J[k] > best_total_J) {
                            best_total_J = total_J[k];
                            centroid_idx = k_idx;
                        }
                    }
                }
                if (centroid_idx != cluster_seed(i)) {
                    cluster_seed(i) = centroid_idx;
                    has_changed = true;
                }
            }
            if (!has_changed) {
                //printf("no change in cluster seeds, breaking!");
                break;
            }
        }
        //printf("\\n");
        py::tuple results(Nall_lengths[0]);
        int n_elem = 0;  // for debugging
        for (int i = 0; i < Ncluster_seed[0]; ++i) {
            for (int j = 0; j < clusters[i].size(); ++j) {
                results[clusters[i][j]] = (int) cluster_seed(i);
                ++n_elem;
            }
        }
        if (n_elem != Nall_lengths[0]) {
            printf("ERROR! number of sets %i but in clustering assigned %i\\n", Nall_lengths[0], n_elem);
        }
        return_val = results;
    '''
    cc_idx = scipy.weave.inline(
        code, ['all_lengths', 'set_limits', 'sets_vector', 'cluster_seed'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        headers=["<vector>"], extra_compile_args=["-O3"]
    )
    assert np.array_equal(np.sort(np.unique(cc_idx)), np.sort(cluster_seed))
    return np.array(cc_idx, dtype=np.int)


def nonnullcols(line):
    return tuple(line.notnull().nonzero()[0])


def compute_feature_sets(filename):
    nchunks = 300
    chunksize = 5000
    set_idx = {}
    hash_dict = {}
    hash_series = {}
    all_responses = []
    for split in ['test', 'train']:
        path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical.csv"
        path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_numeric.csv"
        reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
        reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
        i = 0
        all_hashes = []
        for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
            chunk_categorical = chunk_categorical.set_index('Id')
            chunk_numeric = chunk_numeric.set_index('Id')
            if split == 'train':
                all_responses.append(chunk_numeric['Response'])
                chunk_numeric = chunk_numeric.drop(['Response'], axis=1)  # every record has response, do not use it for grouping
            # assert chunk_categorical.index.equals(chunk_numeric.index)
            whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1)

            feature_indices = whole_chunk.apply(nonnullcols, axis=1)
            hashes = feature_indices.apply(hash)
            all_hashes.append(hashes)
            hash_dict.update(dict(zip(hashes, feature_indices)))
            i += 1
            print split, i
            if i == nchunks:
                break
        hash_series[split] = pd.concat(all_hashes)
        print "Unique ", split, "hashes:", len(hash_series[split].unique())

    response_series = pd.concat(all_responses)
    unique_hashes = pd.concat((hash_series['train'], hash_series['test'])).unique()
    print "Unique combined hashes:", len(unique_hashes)
    unique_feature_sets = [hash_dict[h] for h in unique_hashes]
    for split in ['train', 'test']:
        idx_series = hash_series[split].map(pd.Series(np.arange(len(unique_feature_sets)), index=unique_hashes))
        idx_series = idx_series.rename('feature_set_idx')
        set_idx[split] = idx_series
    set_idx['train'] = pd.concat([set_idx['train'], response_series], axis=1)
    set_idx['test'] = pd.DataFrame(set_idx['test'])
    with open(filename, 'w') as f:
        cPickle.dump((unique_feature_sets, set_idx['train'], set_idx['test']), f, protocol=-1)
    return


def load_feature_sets(filename):
    with open(filename, 'r') as f:
        unique_feature_sets, train_set_idx, test_set_idx = cPickle.load(f)
    return unique_feature_sets, train_set_idx, test_set_idx


def load_maxisets(filename):
    with open(filename, 'r') as f:
        maxisets, train_set_idx, test_set_idx = cPickle.load(f)
    return maxisets, train_set_idx, test_set_idx


def cluster_maxisets(filename, n_clusters):
    unique_feature_sets, train_set_idx, test_set_idx = load_feature_sets('sets.pkl')
    n_sets = None
    if n_sets is not None:
        unique_feature_sets = unique_feature_sets[:n_sets]
        train_set_idx = train_set_idx[train_set_idx['feature_set_idx'] < n_sets]
        test_set_idx = test_set_idx[test_set_idx['feature_set_idx'] < n_sets]

    for _ in range(1):
        cc = jaccard_kmeans(unique_feature_sets, n_clusters)
        train_set_idx['cc'] = cc[train_set_idx['feature_set_idx'].values]
        test_set_idx['cc'] = cc[test_set_idx['feature_set_idx'].values]
        train_record_groups = train_set_idx.groupby(['cc'])
        test_record_groups = test_set_idx.groupby(['cc'])
        train_record_counts = train_record_groups.size()
        test_record_counts = test_record_groups.size()
        all_record_counts = (train_record_counts + test_record_counts).fillna(0)
        response_rate = train_record_groups.mean()['Response']
        n_features_per_record = train_set_idx['feature_set_idx'].apply(lambda x: len(unique_feature_sets[x]))

        g = pd.DataFrame(dict(cc=cc, unique_feature_set=unique_feature_sets)).groupby(['cc'])
        maxisets = g.aggregate({'unique_feature_set': unite})
        maxminmean_J = g.apply(min_max_jaccard)
        maxisets['train_record_count'] = train_record_counts
        maxisets['test_record_count'] = test_record_counts
        maxisets['all_record_count'] = train_record_counts + test_record_counts
        maxisets['response_rate'] = response_rate
        maxisets['feature_count'] = maxisets['unique_feature_set'].apply(lambda x: len(x))
        maxisets = maxisets.join(maxminmean_J)
        record_averaged_J = (maxisets['mean_j'] * all_record_counts).sum() / all_record_counts.sum()
        record_averaged_n_features = n_features_per_record.mean()
        record_averaged_cluster_size = (all_record_counts * all_record_counts).sum() / all_record_counts.sum()
        record_averaged_n_maxifeatures = (maxisets['feature_count'] * all_record_counts).sum() / all_record_counts.sum()
        record_averaged_entropy = (-((1 - response_rate) * np.log(1 - response_rate) +
                                     response_rate * np.log(response_rate + 1e-16)) * train_record_counts).sum() / train_record_counts.sum()
        tot_response_rate = train_set_idx['Response'].mean()
        tot_entropy = -(tot_response_rate * np.log(tot_response_rate) + (1 - tot_response_rate) * np.log(1 - tot_response_rate))
        entropy_gain = (tot_entropy - record_averaged_entropy) / tot_entropy * 100
        print("Record-averaged J: %.3f, record averaged cl size: %.0f, Std/mean record count: %.2f, "
              "record_averaged maxi / n features: %.2f, entropy gain: %.1f%%" %
              (record_averaged_J, record_averaged_cluster_size, all_record_counts.std() / all_record_counts.mean(),
               record_averaged_n_maxifeatures / record_averaged_n_features, entropy_gain))

        print "First 60 sets:"
        print maxisets[['all_record_count', 'train_record_count', 'set_count', 'max_j', 'min_j', 'mean_j', 'feature_count', 'response_rate']].sort_values(by=['all_record_count'])[-60:]

    with open(filename, 'w') as f:
        cPickle.dump((maxisets, train_set_idx, test_set_idx), f, protocol=-1)
    return


def save_all_positives(filename):
    nchunks = 300
    chunksize = 5000
    all_categorical_positives = []
    all_numeric_positives = []
    split = 'train'
    path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical.csv"
    path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_numeric.csv"
    reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
    reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
    i = 0
    for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
        chunk_categorical = chunk_categorical.set_index('Id')
        chunk_numeric = chunk_numeric.set_index('Id')
        positives = chunk_numeric['Response'] > 0
        all_categorical_positives.append(chunk_categorical[positives])
        all_numeric_positives.append(chunk_numeric[positives])
        print i
        i += 1
        if i == nchunks:
            break
    categorical = pd.concat(all_categorical_positives)
    numeric = pd.concat(all_numeric_positives)
    print "Total positives:", len(categorical)
    with open(filename, 'w') as f:
        cPickle.dump((categorical, numeric), f, protocol=-1)
    return


def distribute_data_in_clusters(maxiset_pickle_file):
    maxisets, train_set_idx, test_set_idx = load_maxisets(maxiset_pickle_file)
    set_idx = {'train': train_set_idx, 'test': test_set_idx}
    n_groups = 60
    best_cc = maxisets.sort_values(by='train_record_count').iloc[-n_groups:]
    best_cc['unique_feature_list'] = best_cc['unique_feature_set'].apply(lambda x:list(x))
    nchunks = 300
    chunksize = 5000
    dirname = "clustereddata_" + time.strftime('%y%m%d_%H%M%S', time.localtime())
    os.mkdir(dirname)
    for split in ['train', 'test']:
        path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical.csv"
        path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_numeric.csv"
        reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
        reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
        i = 0
        gdic = defaultdict(list)
        for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
            chunk_categorical = chunk_categorical.set_index('Id')
            chunk_numeric = chunk_numeric.set_index('Id')
            idx_end_categorical = len(chunk_categorical.columns)
            # assert chunk_categorical.index.equals(chunk_numeric.index)
            whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1).join(set_idx[split]['cc'], how='inner')
            if split == 'train':
                response_idx = whole_chunk.columns.get_loc('Response')
            else:
                response_idx = 1e9
            groups = whole_chunk.groupby('cc')
            for cc, group in groups:
                if cc in best_cc.index:
                    feat = np.array(best_cc.loc[cc]['unique_feature_list'], dtype=np.int)
                    categorical_idx = feat[feat < idx_end_categorical]
                    numeric_idx = feat[(feat >= idx_end_categorical) & (feat != response_idx)]
                    datalist = [group.iloc[:,  categorical_idx], group.iloc[:,  numeric_idx]]
                    if split == 'train':
                        datalist.append(group.iloc[:,  response_idx])
                    gdic[cc].append(datalist)
            i += 1
            print split, i
            if i == nchunks:
                break

        del whole_chunk
        del chunk_categorical
        del chunk_numeric
        gc.collect()
        for cc in gdic.keys():
            l = gdic[cc]
            full_categorical = pd.concat([x[0] for x in l])
            full_numeric = pd.concat([x[1] for x in l])
            full = (full_categorical, full_numeric)
            if split == 'train':
                full_response = pd.concat([x[2] for x in l])
                full += (full_response, )
            with open(os.path.join(dirname, 'group_%s_%i.pkl' % (split, cc)), 'w') as f:
                cPickle.dump(full, f, protocol=-1)
            print split, cc, len(full_numeric), len(full_categorical.columns), len(full_numeric.columns)
            del gdic[cc]
            gc.collect()
        del(gdic)
        gc.collect()


def _make_stats_numeric(data):
    return pd.DataFrame({
        # 'max': data.max(axis=0),
        # 'min': data.min(axis=0),
        # 'median': data.median(axis=0),
        'presence': data.count(axis=0) / float(len(data)),
        'mean': data.mean(axis=0),
        'std': data.std(axis=0)})

def _make_stats_categorical(data):
    return pd.DataFrame({
        # 'max': data.max(axis=0),
        # 'min': data.min(axis=0),
        # 'median': data.median(axis=0),
        'presence': data.count(axis=0) / float(len(data)),
        'unique': data.apply(lambda x: len(x.unique()) - int(x.isnull().values.any()))})


def _matthews_correlation(truth, x):
    TP = np.sum(truth * x)
    TN = np.sum((1 - truth) * (1 - x))
    FP = np.sum((1 - truth) * x)
    FN = np.sum(truth * (1 - x))
    assert np.abs(TP + FP + TN + FN - len(truth) < 1e-6)
    score = ((TP * TN) - (FP * FN)) / np.sqrt(1e-12 + (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return score, TP, TN, FP, FN


def _predict_cluster(train_categorical, train_numeric, train_response, test_categorical, test_numeric,
                     other_positives_categorical, other_positives_numeric, remove_absent=True, remove_invariant=True):
    categorical_stats = _make_stats_categorical(train_categorical)
    numeric_stats = _make_stats_numeric(train_numeric)
    # print categorical_stats
    # print numeric_stats
    # print "numeric:", len(train_numeric.columns), "categorical:", len(train_categorical.columns)
    if remove_absent:
        present_categorical = (categorical_stats['presence'] > 0.001)
        present_numeric = numeric_stats['presence'] > 0.50
        train_categorical = train_categorical.loc[:, present_categorical]
        train_numeric = train_numeric.loc[:, present_numeric]
        test_categorical = test_categorical.loc[:, present_categorical]
        test_numeric = test_numeric.loc[:, present_numeric]
        other_positives_categorical = other_positives_categorical.loc[:, present_categorical]
        other_positives_numeric = other_positives_numeric.loc[:, present_numeric]
        categorical_stats = _make_stats_categorical(train_categorical)
        numeric_stats = _make_stats_numeric(train_numeric)
        # print "Removed absents:"
        # print categorical_stats
        # print numeric_stats

    # print "After absents numeric:", len(train_numeric.columns), "categorical:", len(train_categorical.columns)
    # Always remove numeric features that have no variation in the training set
    variant_numeric = (numeric_stats['std'] > 0)
    train_numeric = train_numeric.loc[:, variant_numeric]
    test_numeric = test_numeric.loc[:, variant_numeric]
    other_positives_numeric = other_positives_numeric.loc[:, variant_numeric]
    numeric_stats = _make_stats_numeric(train_numeric)
    if remove_invariant:
        variant_categorical = (categorical_stats['unique'] > 2)
        train_categorical = train_categorical.loc[:, variant_categorical]
        test_categorical = test_categorical.loc[:, variant_categorical]
        other_positives_categorical = other_positives_categorical.loc[:, variant_categorical]
        categorical_stats = _make_stats_categorical(train_categorical)
        # print "Removed invariant:"
        # print categorical_stats
        # print numeric_stats

    # print "After invariants numeric:", len(train_numeric.columns), "categorical:", len(train_categorical.columns)
    # Time to select meaningful positives and join them to the training set
    categorical_informative_features = other_positives_categorical.count(axis=1)
    numeric_informative_features = other_positives_numeric.count(axis=1)
    informative_positives = (categorical_informative_features >= 0.1 * categorical_stats['presence'].sum()) &\
                            (numeric_informative_features > 0.8 * numeric_stats['presence'].sum())
    other_positives_categorical = other_positives_categorical[informative_positives]
    other_positives_numeric = other_positives_numeric[informative_positives]
    train_categorical = pd.concat((train_categorical, other_positives_categorical))
    train_numeric = pd.concat((train_numeric, other_positives_numeric))
    train_response = pd.concat((train_response, pd.Series(np.ones(len(other_positives_categorical)),
                                                          index=other_positives_categorical.index.values, name='Response')))
    train_present = (train_categorical.count(axis=1) + train_numeric.count(axis=1)) / (len(train_categorical.columns) + len(train_numeric.columns))

    # Notice now we use both training and testing to get the mean to replace nans
    numeric_stats = _make_stats_numeric(pd.concat((train_numeric, test_numeric), axis=0))
    # Fill numeric nans with the training + testing mean
    train_numeric = train_numeric.fillna(numeric_stats['mean'], axis=0)
    test_numeric = test_numeric.fillna(numeric_stats['mean'], axis=0)

    # Transform categorical data into consecutive numbers
    all_categorical = pd.concat((train_categorical, test_categorical), axis=0)
    all_categorical = all_categorical.rank(axis=0, method='dense', na_option='bottom')
    train_ranked_categorical = all_categorical.iloc[:len(train_categorical)]
    test_ranked_categorical = all_categorical.iloc[len(train_categorical):]

    # numeric_stats = _make_stats_numeric(train_numeric)
    # categorical_stats = _make_stats_categorical(train_ranked_categorical_data)
    # print "Filled nans:"
    # print categorical_stats
    # print numeric_stats

    all_train_data = pd.concat((train_ranked_categorical, train_numeric), axis=1)
    all_test_data = pd.concat((test_ranked_categorical, test_numeric), axis=1)
    all_stats = _make_stats_numeric(pd.concat((all_train_data, all_test_data), axis=0))
    all_train_data = (all_train_data - all_stats['mean']) / (1e-9 + all_stats['std'])
    all_test_data = (all_test_data - all_stats['mean']) / (1e-9 + all_stats['std'])

    # all_stats = _make_stats_numeric(all_train_data)
    # print "Normalized:"
    # print all_stats

    positive_rate = np.mean(train_response)
    weighted_positive_rate = np.sum(train_present * train_response) / np.sum(train_present * (1 - train_response))
    sample_weights = train_present / weighted_positive_rate * train_response + train_present * (1 - train_response)
    train_resample = pd.concat((all_train_data, train_response), axis=1).sample(frac=0.75, replace=True, weights=sample_weights)
    resampled_response = train_resample['Response']
    train_resample = train_resample.drop('Response', axis=1)
    # print("Added %i other positives, resampled positives from %.3f to %.3f positives" % (len(other_positives_categorical), positive_rate, resampled_response.mean()))

    parts = []
    results = []
    for l in ['L']:
        features = [s for s in train_resample.columns.values if s[:1] == l]
        if len(features) > 0:
            parts.append(features)
    print "part lengths:", [len(s) for s in parts]
    for p in parts:
        # clf = SVC(C=0.05, kernel='linear')
        # clf = LinearSVC(C=0.01, penalty='l1', dual=False, class_weight='balanced')
        clf = RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=20, class_weight='balanced')
        # clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=8)
        clf2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'), n_estimators=50)
        # clf2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
        clf2 = RadiusNeighborsClassifier()
        clf.fit(train_resample.loc[:, p], resampled_response)
        fi = clf.feature_importances_
        clf2.fit(train_resample.loc[:, p] * fi, resampled_response)
        results.append(clf2.predict(all_test_data.loc[:, p] * fi))
    result = np.sum(np.vstack(results), axis=0) > 0
    return result


def main():
    dirname = 'clustereddata_300_best50'
    with open('positives.pkl', 'r') as f:
        all_positives_categorical, all_positives_numeric = cPickle.load(f)

    all_files = os.listdir(dirname)
    all_train_files = [s for s in all_files if 'train' in s]
    sizes = [os.path.getsize(os.path.join(dirname, s)) for s in all_train_files]
    all_idx = np.argsort(sizes)[::-1]  # biggest first
    all_TP = all_TN = all_FP = all_FN = 0
    tot_records = 0
    for idx in all_idx[40:45]:
        train_file = os.path.join(dirname, all_train_files[idx])
        test_file = train_file.replace('train', 'test')
        with open(train_file, 'r') as f:
            train_categorical, train_numeric, train_response = cPickle.load(f)
        tot_records += len(train_categorical)
        print("Processing %s, size=%i, n_records=%i" % (all_train_files[idx], sizes[idx], len(train_categorical)))
        others = ~all_positives_categorical.index.isin(train_categorical.index)
        other_positives_categorical = all_positives_categorical.loc[others]
        other_positives_numeric = all_positives_numeric.loc[others]
        other_positives_categorical = other_positives_categorical[train_categorical.columns]
        other_positives_numeric = other_positives_numeric[train_numeric.columns]
        # print("Kept %i extra positives out of %i" % (len(other_positives_categorical), len(all_positives_categorical)))
        # with open(test_file, 'r') as f:
        #     test_categorical, test_numeric = cPickle.load(f)

        kf = KFold(len(train_response), n_folds=5)
        remove_absent = True
        remove_invariant = True
        pd.options.display.float_format = '{:,.2f}'.format
        for train, test in kf:
            train_categorical_chunk = train_categorical.iloc[train]
            train_numeric_chunk = train_numeric.iloc[train]
            train_response_chunk = train_response.iloc[train]
            validation_categorical = train_categorical.iloc[test]
            validation_numeric = train_numeric.iloc[test]
            validation_response = train_response.iloc[test]
            response = _predict_cluster(train_categorical_chunk, train_numeric_chunk, train_response_chunk,
                                        validation_categorical, validation_numeric,
                                        other_positives_categorical, other_positives_numeric,
                                        remove_absent=remove_absent, remove_invariant=remove_invariant)
            score, TP, TN, FP, FN = _matthews_correlation(validation_response, response)
            print "Score: %.3f, positives %i, true %i, false %i" % (score, np.sum(validation_response), TP, FP)
            all_TP += TP
            all_TN += TN
            all_FP += FP
            all_FN += FN
        print "Cum records:", tot_records, "Cumulative score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
            np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))

    print "Final score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
                          np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))


if __name__ == "__main__":
    # cluster_maxisets('maxisets_600.pkl', 600)
    # distribute_data_in_clusters('maxisets_600.pkl')

    main()
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
