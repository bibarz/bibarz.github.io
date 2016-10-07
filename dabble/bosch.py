import numpy as np
import pandas as pd
import itertools
import scipy.weave
import cPickle
import os
import time
import xgboost
import gc
from collections import defaultdict
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, TomekLinks
from athena import Convnet


def check_feature_equality():
    nchunks = 300
    chunksize = 5000
    fpath = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/feature_summary.csv"
    fsum = pd.read_csv(fpath, header=0)
    groups = fsum.groupby('digest')
    for split in ['train', 'test']:
        checked_cols = False
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        for chunk in reader:
            if not checked_cols:
                assert (chunk.columns==fsum['feature']).all()
                checked_cols = True
            for digest, g in groups:
                columns = g.feature
                for c in columns[1:]:
                    if not chunk.loc[:, columns.iloc[0]].equals(chunk.loc[:, c]):
                        different = chunk.loc[:, columns.iloc[0]].fillna(-9999) != chunk.loc[:, c].fillna(-9999)
                        print "col %s differs from col %s: %s, %s" % (columns.iloc[0], c, chunk.loc[different, columns.iloc[0]], chunk.loc[different, c])
            i += 1
            print i
            if i == nchunks:
                break

def clean_categorical():
    nchunks = 300
    chunksize = 5000
    fpath = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/feature_summary.csv"
    fsum = pd.read_csv(fpath, header=0)
    cols = fsum.feature[~fsum.duplicate]
    print cols
    for split in ['train', 'test']:
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical_dirty.csv"
        write_path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_categorical_clean.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        for chunk in reader:
            exists = os.path.exists(write_path)
            nondupes = chunk.loc[:, cols]
            nondupes = nondupes.set_index('Id')
            if not exists:
                print "from %i to %i cols" % (chunk.shape[1], nondupes.shape[1])
            with open(write_path, 'a') as f:
                nondupes.to_csv(f, sep=',', header=not exists, na_rep='')
            i += 1
            print i
            if i == nchunks:
                break


def examine_date():
    nchunks = 30
    chunksize = 50
    for split in ['train', 'test']:
        grouped = False
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        for chunk in reader:
            if not grouped:
                cols = chunk.columns
                st = cols.str.rsplit('_', n=1).str[0]
                groups = chunk.columns.groupby(st)
                grouped = True
            for station, columns in groups.iteritems():
                for c in columns[1:]:
                    if not chunk.loc[:, columns[0]].equals(chunk.loc[:, c]):
                        different = chunk.loc[:, columns[0]].fillna(-9999) != chunk.loc[:, c].fillna(-9999)
                        print "col %s differs from col %s: %s, %s" % (columns[0], c, chunk.loc[different, columns[0]].values,
                                                                      chunk.loc[different, c].values)
            i += 1
            print i
            if i == nchunks:
                break


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
        paths = ["/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + ("_%s.csv" % (k, ))
                 for k in ['categorical', 'date', 'numeric']]
        readers = [pd.read_table(p, chunksize=chunksize, header=0, sep=',') for p in paths]
        i = 0
        all_hashes = []
        for chunks in itertools.izip(*readers):
            chunks = [c.set_index('Id') for c in chunks]
            if split == 'train':
                all_responses.append(chunks[2]['Response'])  # chunks[2] is numeric
                chunks[2] = chunks[2].drop(['Response'], axis=1)  # every record has response, do not use it for grouping
            assert chunks[0].index.equals(chunks[1].index)
            # assert chunks[0].index.equals(chunks[2].index)
            whole_chunk = pd.concat(chunks, axis=1)

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
        maxisets['all_record_count'] = all_record_counts
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

        print "All sets:"
        pd.options.display.max_rows = 9999
        print maxisets[['all_record_count', 'train_record_count', 'set_count', 'max_j', 'min_j', 'mean_j', 'feature_count', 'response_rate']].sort_values(by=['all_record_count'])

    with open(filename, 'w') as f:
        cPickle.dump((maxisets, train_set_idx, test_set_idx), f, protocol=-1)
    return


def save_all_positives(filename):
    nchunks = 300
    chunksize = 5000
    keys = ['categorical', 'date', 'numeric']
    all_positives = [[] for _ in keys]
    split = 'train'
    paths = ["/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + ("_%s.csv" % (k, ))
             for k in keys]
    readers = [pd.read_table(p, chunksize=chunksize, header=0, sep=',') for p in paths]
    i = 0
    for chunks in itertools.izip(*readers):
        chunks = [c.set_index('Id') for c in chunks]
        positives = chunks[2]['Response'] > 0
        for j, k in enumerate(keys):
            all_positives[j].append(chunks[j][positives])
        print "Chunk", i
        i += 1
        if i == nchunks:
            break
    all_positive_frames = tuple([pd.concat(c) for c in all_positives])
    print "Total positives:", len(all_positive_frames[0])
    with open(filename, 'w') as f:
        cPickle.dump(all_positive_frames, f, protocol=-1)
    return


def dump_to_csv_cluster(dirname, cc, split, datadict):
    for key, frame in datadict.iteritems():
        path = os.path.join(dirname, 'group_%s_%s_%.6d.csv' % (split, key, cc))
        exists = os.path.exists(path)
        with open(path, 'a') as f:
            frame.to_csv(f, sep=',', header=not exists, na_rep='')


def distribute_data_in_clusters(maxiset_pickle_file):
    maxisets, train_set_idx, test_set_idx = load_maxisets(maxiset_pickle_file)
    set_idx = {'train': train_set_idx, 'test': test_set_idx}
    n_groups = 0  # 0 meaning all
    best_cc = maxisets.sort_values(by='train_record_count').iloc[-n_groups:]
    best_cc['unique_feature_list'] = best_cc['unique_feature_set'].apply(lambda x:list(x))
    nchunks = 300
    chunksize = 5000
    dirname = "clustereddata_" + time.strftime('%y%m%d_%H%M%S', time.localtime())
    os.mkdir(dirname)
    keys = ['categorical', 'date', 'numeric']
    for split in ['train', 'test']:
        paths = ["/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + ("_%s.csv" % (k, ))
                 for k in keys]
        readers = [pd.read_table(p, chunksize=chunksize, header=0, sep=',') for p in paths]
        i = 0
        for chunks in itertools.izip(*readers):
            chunks = [c.set_index('Id') for c in chunks]
            idx_end_categorical = len(chunks[0].columns)
            idx_end_date = len(chunks[0].columns) + len(chunks[1].columns)
            assert chunks[0].index.equals(chunks[1].index)
            whole_chunk = pd.concat(chunks, axis=1).join(set_idx[split]['cc'], how='inner')
            if split == 'train':
                response_idx = whole_chunk.columns.get_loc('Response')
            else:
                response_idx = 1e9
            groups = whole_chunk.groupby('cc')
            for cc, group in groups:
                if cc in best_cc.index:
                    feat = np.array(best_cc.loc[cc]['unique_feature_list'], dtype=np.int)
                    categorical_idx = feat[feat < idx_end_categorical]
                    date_idx = feat[(feat >= idx_end_categorical) & (feat < idx_end_date)]
                    numeric_idx = feat[(feat >= idx_end_date) & (feat != response_idx)]
                    datadict = {'categorical': group.iloc[:,  categorical_idx],
                                'date': group.iloc[:,  date_idx],
                                'numeric': group.iloc[:,  numeric_idx]}
                    if split == 'train':
                        datadict['response'] = group.iloc[:,  response_idx]
                    dump_to_csv_cluster(dirname, cc, split, datadict)
            i += 1
            print split, i
            if i == nchunks:
                break


def _make_stats(key, data):
    if key == 'numeric':
        return pd.DataFrame({
            # 'max': data.max(axis=0),
            # 'min': data.min(axis=0),
            # 'median': data.median(axis=0),
            'presence': data.count(axis=0) / float(len(data)),
            'mean': data.mean(axis=0),
            'std': data.std(axis=0)})
    if key == 'date':
        return pd.DataFrame({
            # 'max': data.max(axis=0),
            # 'min': data.min(axis=0),
            # 'median': data.median(axis=0),
            'presence': data.count(axis=0) / float(len(data)),
            'mean': data.mean(axis=0),
            'std': data.std(axis=0)})
    if key == 'categorical':
        return pd.DataFrame({
            # 'max': data.max(axis=0),
            # 'min': data.min(axis=0),
            # 'median': data.median(axis=0),
            'presence': data.count(axis=0) / float(len(data)),
            'unique': data.apply(lambda x: len(x.unique()) - int(x.isnull().values.any()))})
    if key == 'response':
        return pd.DataFrame({
            'mean': data.mean(axis=0)})
    raise Exception('Stats of what? Key %s not recognized' % (key, ))


def _matthews_correlation(truth, x):
    TP = np.sum(truth * x)
    TN = np.sum((1 - truth) * (1 - x))
    FP = np.sum((1 - truth) * x)
    FN = np.sum(truth * (1 - x))
    assert np.abs(TP + FP + TN + FN - len(truth) < 1e-6)
    score = ((TP * TN) - (FP * FN)) / np.sqrt(1e-12 + (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return score, TP, TN, FP, FN


def get_xgb_clf():
    params = {}
    params['objective'] = "binary:logistic"
    params['learning_rate'] = 0.05
    params['max_depth'] = 10
    params['n_estimators'] = 500
    # params['colsample_bytree'] = 0.82
    params['min_child_weight'] = 3
    params['base_score'] = 0.005
    params['silent'] = True
    clf = xgboost.XGBClassifier(**params)
    return clf


def _predict_cluster(train_dict, test_dict, other_positives_dict,
                     remove_absent=True, remove_invariant=True):
    '''
    :param train_dict: with keys 'categorical', 'date', 'numeric' and 'response', each a frame
    :param test_dict: with keys 'categorical', 'date', and 'numeric', each a frame
    :param other_positives_dict: with keys 'categorical', 'date', and 'numeric', each a frame
    '''
    stats = {k: _make_stats(k, v) for k, v in train_dict.iteritems()}
    # print stats['categorical']
    # print stats['date']
    # print stats['numeric']
    # print stats['response']
    # print "numeric:", len(train_dict['numeric'].columns), "date:", len(train_dict['date'].columns), "categorical:", len(train_dict['categorical'].columns)
    if remove_absent:
        present_dict = {k: stats[k]['presence'] > thr for k, thr in
                        [('categorical', 0.001), ('date', 0.50), ('numeric', 0.50)]}
        for k in present_dict.keys():
            train_dict[k] = train_dict[k].loc[:, present_dict[k]]
            test_dict[k] = test_dict[k].loc[:, present_dict[k]]
            other_positives_dict[k] = other_positives_dict[k].loc[:, present_dict[k]]
            stats[k] = _make_stats(k, train_dict[k])
            # print "Removed absents:"
            # print stats['categorical']
            # print stats['date']
            # print stats['numeric']
            # print stats['response']

    # print "After absents numeric:", len(train_dict['numeric'].columns), "date:", len(train_dict['date'].columns), "categorical:", len(train_dict['categorical'].columns)
    # Always remove numeric and date features that have no variation in the training set
    # Categorical gets special conditions
    variant_conditions = [('numeric', 'std', 0.), ('date', 'std', 0.)]
    if remove_invariant:
        variant_conditions.append(('categorical', 'unique', 2))
    for k, stat, thr in variant_conditions:
        variant = (stats[k][stat] > thr)
        train_dict[k] = train_dict[k].loc[:, variant]
        test_dict[k] = test_dict[k].loc[:, variant]
        other_positives_dict[k] = other_positives_dict[k].loc[:, variant]
        stats[k] = _make_stats(k, train_dict[k])
    # print "Removed invariant:"
    # print stats['categorical']
    # print stats['date']
    # print stats['numeric']

    # print "After invariants numeric:", len(train_dict['numeric'].columns), "date:", len(train_dict['date'].columns), "categorical:", len(train_dict['categorical'].columns)
    # Time to select meaningful positives and join them to the training set
    informations = [('categorical', 0.1), ('date', 0.5), ('numeric', 0.8)]
    informative_positives_by_type = [other_positives_dict[k].count(axis=1) >= thr * stats[k]['presence'].sum()
                                     for k, thr in informations]
    informative_positives = pd.concat(informative_positives_by_type, axis=1).all(axis=1)
    for k in ['categorical', 'date', 'numeric']:
        other_positives_dict[k] = other_positives_dict[k][informative_positives]
        train_dict[k] = pd.concat((train_dict[k], other_positives_dict[k]))
    train_dict['response'] = pd.concat((train_dict['response'], pd.DataFrame({'Response':np.ones(len(other_positives_dict['categorical'])),
                                                                              'Id': other_positives_dict['categorical'].index.values}).set_index('Id')))

    # use diffs of times
    # train_dict['date'] = train_dict['date'].sub(train_dict['date'].median(axis=1).fillna(0), axis=0)
    # test_dict['date'] = test_dict['date'].sub(test_dict['date'].median(axis=1).fillna(0), axis=0)
    train_date_median = train_dict['date'].median(axis=1)
    test_date_median = test_dict['date'].median(axis=1)
    train_dict['date'] = train_dict['date'].diff(axis=1)
    test_dict['date'] = test_dict['date'].diff(axis=1)
    train_dict['date'].iloc[:, 0] = train_date_median  # replace the first column of diff, which is all nans, with the median
    test_dict['date'].iloc[:, 0] = test_date_median  # replace the first column of diff, which is all nans, with the median

    # Notice now we use both training and testing to get the mean to replace nans
    for k in ['date', 'numeric']:
        stats[k] = _make_stats(k, pd.concat((train_dict[k], test_dict[k]), axis=0))
        # Fill numeric and date nans with the training + testing mean
        train_dict[k] = train_dict[k].fillna(stats[k]['mean'], axis=0)
        test_dict[k] = test_dict[k].fillna(stats[k]['mean'], axis=0)

    # Transform categorical data into consecutive numbers
    all_categorical = pd.concat((train_dict['categorical'], test_dict['categorical']), axis=0)
    all_categorical = all_categorical.rank(axis=0, method='dense', na_option='bottom')
    test_dict['categorical'] = all_categorical.iloc[len(train_dict['categorical']):]
    train_dict['categorical'] = all_categorical.iloc[:len(train_dict['categorical'])]

    # for k in train_dict.keys():
    #     stats[k] = _make_stats(k, train_dict[k])
    # print "Filled nans:"
    # print stats['categorical']
    # print stats['date']
    # print stats['numeric']

    all_train_data = pd.concat((train_dict[k] for k in ['categorical', 'date', 'numeric']), axis=1)
    all_test_data = pd.concat((test_dict[k] for k in ['categorical', 'date', 'numeric']), axis=1)
    all_stats = _make_stats('numeric', pd.concat((all_train_data, all_test_data), axis=0))
    all_train_data = (all_train_data - all_stats['mean']) / (1e-9 + all_stats['std'])
    all_test_data = (all_test_data - all_stats['mean']) / (1e-9 + all_stats['std'])

    positives = all_train_data[(train_dict['response'] == 1).values]
    negatives = all_train_data[(train_dict['response'] == 0).values]
    nn = NearestNeighbors(n_neighbors=40).fit(negatives)
    d, nearest = nn.kneighbors(positives)
    nearest = np.unique(nearest.ravel())
    all_train_data = pd.concat((positives, negatives.iloc[nearest.ravel()]))
    r = np.concatenate((np.ones(len(positives)), np.zeros(len(nearest.ravel()))))

    resampled_response = r
    train_resample = all_train_data
    # pca, _ = pca_converter(train_resample, 1, 0.9)
    # all_stats = _make_stats('numeric', all_train_data)
    # print "Normalized:"
    # print all_stats

    # r = train_dict['response'].values.ravel()
    positive_rate = np.mean(r)
    train_present = (all_train_data.count(axis=1)) / float(len(all_train_data.columns))

    # oversampler = SMOTE()
    # train_resample, resampled_response = oversampler.fit_sample(all_train_data.values, train_dict['response'].values.ravel())
    # train_resample = pd.DataFrame(train_resample, columns=all_train_data.columns)
    weighted_positive_rate = np.sum(train_present * r) / np.sum(train_present * (1 - r))
    sample_weights = train_present / weighted_positive_rate * r + train_present * (1 - r)
    train_resample = pd.concat((all_train_data, pd.DataFrame({'Response': r, 'Id': all_train_data.index.values}).set_index('Id')), axis=1).sample(frac=1.0, replace=True, weights=sample_weights)
    resampled_response = train_resample['Response']
    train_resample = train_resample.drop('Response', axis=1)
    print("Added %i other positives, resampled positives from %.3f to %.3f positives; total samples from %i to %i, %i features" %
          (len(other_positives_dict['categorical']), positive_rate, resampled_response.mean(),
           len(all_train_data), len(train_resample), len(train_resample.columns)))

    parts = []
    results = []
    for l in ['L']:
        features = [s for s in train_resample.columns.values if True]#s[:1] == l]
        if len(features) > 0:
            parts.append(features)
    # print "part lengths:", [len(s) for s in parts]
    for p in parts:
        # clf = SVC(C=0.05, kernel='linear')
        # clf = LinearSVC(C=0.01, penalty='l1', dual=False, class_weight='balanced')
        clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=12,
                                     max_features=min(len(train_resample.columns), int(4 * np.sqrt(len(train_resample.columns)))),
                                     class_weight='balanced')
        the_train_data = train_resample.loc[:, p]
        the_test_data = all_test_data.loc[:, p]
        # the_train_data = pca(the_train_data)
        # the_test_data = pca(the_test_data)
        # clf = Convnet(data_length=the_data.shape[1], n_channels=1, n_outputs=2,
        #               kernel_defs=[(1, 4, 1)], fc_sizes=[16],
        #               batch_size=128, n_passes=60, dropout_keep_prob=0.5)
        clf = get_xgb_clf()
        # clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1., min_samples_leaf=5, max_depth=10)
        # clf2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'), n_estimators=50)
        # clf2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
        # clf2 = RadiusNeighborsClassifier()
        clf.fit(the_train_data, resampled_response)
        # fi = clf.feature_importances_
        # clf2.fit(train_resample.loc[:, p] * fi, resampled_response)
        # results.append(clf2.predict(all_test_data.loc[:, p] * fi))
        results.append(clf.predict(the_test_data))
    result = np.sum(np.vstack(results), axis=0) > 0
    return result.astype(np.float)


def main():
    dirname = 'clustereddata_300_csv'
    all_positives = {}
    with open('positives.pkl', 'r') as f:
        all_positives['categorical'], all_positives['date'], all_positives['numeric'] = cPickle.load(f)

    all_files = os.listdir(dirname)
    train_files = {}
    test_files = {}
    for k in ['categorical', 'date', 'numeric', 'response']:
        train_files[k] = [s for s in all_files if k in s and 'train' in s]
    for k in ['categorical', 'date', 'numeric']:
        test_files[k] = [s for s in all_files if k in s and 'test' in s]
    sizes = [os.path.getsize(os.path.join(dirname, s)) for s in train_files['numeric']]
    all_idx = np.argsort(sizes)[::-1]  # biggest first
    all_TP = all_TN = all_FP = all_FN = 0
    tot_records = 0
    for idx in all_idx[7:10]:
        cc = train_files['numeric'][idx][-10:-4]
        train_dict = {}
        test_dict = {}
        for k in ['categorical', 'date', 'numeric', 'response']:
            train_file = os.path.join(dirname, [s for s in train_files[k] if cc in s][0])
            train_dict[k] = pd.read_csv(train_file, sep=',', header=0).set_index('Id')
        for k in ['categorical', 'date', 'numeric']:
            test_file = os.path.join(dirname, [s for s in test_files[k] if cc in s][0])
            test_dict[k] = pd.read_csv(test_file, sep=',', header=0).set_index('Id')
        tot_records += len(train_dict['categorical'])
        print("Processing %s, size=%i, n_records=%i" % (cc, sizes[idx], len(train_dict['categorical'])))
        others = ~all_positives['categorical'].index.isin(train_dict['categorical'].index)
        other_positives_dict = {k: all_positives[k].loc[others, train_dict[k].columns] for k in all_positives.keys()}
        # print("Kept %i extra positives out of %i" % (len(other_positives['categorical']), len(all_positives['categorical'])))

        kf = KFold(len(train_dict['response']), n_folds=5)
        remove_absent = True
        remove_invariant = True
        pd.options.display.float_format = '{:,.2f}'.format
        for train, test in kf:
            train_chunk_dict = {k: train_dict[k].iloc[train] for k in train_dict.keys()}
            validation_dict = {k: train_dict[k].iloc[test] for k in train_dict.keys()}
            response = _predict_cluster(train_chunk_dict, validation_dict, other_positives_dict,
                                        remove_absent=remove_absent, remove_invariant=remove_invariant)
            score, TP, TN, FP, FN = _matthews_correlation(validation_dict['response'].values.ravel(), response)
            print "Score: %.3f, positives %i, true %i, false %i" % (score, np.sum(validation_dict['response']), TP, FP)
            all_TP += TP
            all_TN += TN
            all_FP += FP
            all_FN += FN
        print "Cum records:", tot_records, "Cumulative score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
            np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))

    print "Final score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
                          np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))


if __name__ == "__main__":
    # save_all_positives('kk.pkl')
    # compute_feature_sets('new_sets.pkl')
    # cluster_maxisets('maxisets_300.pkl', 300)
    # distribute_data_in_clusters('maxisets_300.pkl')
    # clean_categorical()
    # examine_date()
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
