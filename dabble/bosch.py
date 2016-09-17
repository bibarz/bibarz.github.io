import numpy as np
import pandas as pd
import itertools
import scipy.weave
import cPickle
from collections import defaultdict


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
    path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_categorical.csv"
    path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_numeric.csv"
    reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
    reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
    i = 0
    all_hashes = []
    all_responses = []
    all_stats = []
    hash_dict = {}
    for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
        chunk_categorical.set_index('Id', inplace=True)
        chunk_numeric.set_index('Id', inplace=True)
        all_responses.append(chunk_numeric['Response'])
        chunk_numeric.drop(['Response'], axis=1, inplace=True)  # every record has response, do not use it for grouping
        # assert chunk_categorical.index.equals(chunk_numeric.index)
        whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1)
        all_stats.append(pd.DataFrame({'count': whole_chunk.count(axis=0),
                                       'sum': whole_chunk.fillna(0).sum(axis=0),
                                       'max': whole_chunk.fillna(-1e9).max(axis=0),
                                       'min': whole_chunk.fillna(1e9).min(axis=0),
                                       'sum_sq': whole_chunk.fillna(0).apply(lambda x: x**2).sum(axis=0)}))

        feature_indices = whole_chunk.apply(nonnullcols, axis=1)
        hashes = feature_indices.apply(hash)
        all_hashes.append(hashes)
        hash_dict.update(dict(zip(hashes, feature_indices)))
        i += 1
        print i
        if i == nchunks:
            break
    stats = pd.DataFrame({'count': pd.concat([f['count'] for f in all_stats], axis=1).sum(axis=1),
                          'sum': pd.concat([f['sum'] for f in all_stats], axis=1).sum(axis=1),
                          'sum_sq': pd.concat([f['sum_sq'] for f in all_stats], axis=1).sum(axis=1),
                          'max': pd.concat([f['max'] for f in all_stats], axis=1).max(axis=1),
                          'min': pd.concat([f['min'] for f in all_stats], axis=1).min(axis=1)})

    hash_series = pd.concat(all_hashes)
    unique_hashes = hash_series.unique()
    unique_feature_sets = [hash_dict[h] for h in unique_hashes]
    idx_series = hash_series.map(pd.Series(np.arange(len(unique_feature_sets)), index=unique_hashes))
    idx_series = idx_series.rename('feature_set_idx')
    response_series = pd.concat(all_responses)
    set_and_response = pd.concat([idx_series, response_series], axis=1)
    with open(filename, 'w') as f:
        cPickle.dump((unique_feature_sets, set_and_response, stats), f, protocol=-1)
    return


def load_feature_sets(filename):
    with open(filename, 'r') as f:
        unique_feature_sets, set_and_response, stats = cPickle.load(f)
    return unique_feature_sets, set_and_response, stats


def load_maxisets(filename):
    with open(filename, 'r') as f:
        maxisets, set_and_response = cPickle.load(f)
    return maxisets, set_and_response


def cluster_maxisets(filename):
    unique_feature_sets, set_and_response, stats = load_feature_sets('sets.pkl')
    n_sets = None
    n_clusters = 300
    if n_sets is not None:
        unique_feature_sets = unique_feature_sets[:n_sets]
        set_and_response = set_and_response[set_and_response['feature_set_idx'] < n_sets]

    for _ in range(1):
        cc = jaccard_kmeans(unique_feature_sets, n_clusters)

        set_and_response['cc'] = cc[set_and_response['feature_set_idx'].values]
        record_groups = set_and_response.groupby(['cc'])
        record_counts = record_groups.size()
        response_rate = record_groups.mean()['Response']
        n_features_per_record = set_and_response['feature_set_idx'].apply(lambda x: len(unique_feature_sets[x]))

        g = pd.DataFrame(dict(cc=cc, unique_feature_set=unique_feature_sets)).groupby(['cc'])
        maxisets = g.aggregate({'unique_feature_set': unite})
        maxminmean_J = g.apply(min_max_jaccard)
        maxisets['record_count'] = record_counts
        maxisets['response_rate'] = response_rate
        maxisets['feature_count'] = maxisets['unique_feature_set'].apply(lambda x: len(x))
        maxisets = maxisets.join(maxminmean_J)
        record_averaged_J = (maxisets['mean_j'] * record_counts).sum() / record_counts.sum()
        record_averaged_n_features = n_features_per_record.mean()
        record_averaged_cluster_size = (record_counts * record_counts).sum() / record_counts.sum()
        record_averaged_n_maxifeatures = (maxisets['feature_count'] * record_counts).sum() / record_counts.sum()
        record_averaged_entropy = (-((1 - response_rate) * np.log(1 - response_rate) +
                                     response_rate * np.log(response_rate + 1e-16)) * record_counts).sum() / record_counts.sum()
        tot_response_rate = set_and_response['Response'].mean()
        tot_entropy = -(tot_response_rate * np.log(tot_response_rate) + (1 - tot_response_rate) * np.log(1 - tot_response_rate))
        entropy_gain = (tot_entropy - record_averaged_entropy) / tot_entropy * 100
        print("Record-averaged J: %.3f, record averaged cl size: %.0f, Std/mean record count: %.2f, "
              "record_averaged maxi / n features: %.2f, entropy gain: %.1f%%" %
              (record_averaged_J, record_averaged_cluster_size, record_counts.std() / record_counts.mean(),
               record_averaged_n_maxifeatures / record_averaged_n_features, entropy_gain))

        # print "First 30 sets:"
        # print maxisets[['record_count', 'set_count', 'max_j', 'min_j', 'mean_j', 'feature_count', 'response_rate']].sort_values(by=['record_count'])[-30:]
        # print maxisets[['record_count', 'set_count', 'max_j', 'min_j', 'mean_j', 'feature_count', 'response_rate']].sort_values(by=['record_count'])

    with open(filename, 'w') as f:
        cPickle.dump((maxisets, set_and_response), f, protocol=-1)
    return


def main():
    maxisets, set_and_response = load_maxisets('maxisets_300.pickle')
    n_groups = 10
    best_cc = maxisets.sort_values(by='record_count').iloc[-n_groups:]
    maxisets['unique_feature_list'] = maxisets['unique_feature_set'].apply(lambda x:list(x))
    nchunks = 300
    chunksize = 5000
    path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_categorical.csv"
    path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_numeric.csv"
    reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
    reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
    i = 0
    gdic = defaultdict(list)
    for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
        chunk_categorical.set_index('Id', inplace=True)
        chunk_numeric.set_index('Id', inplace=True)
        # assert chunk_categorical.index.equals(chunk_numeric.index)
        whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1).join(set_and_response['cc'], how='inner')
        response_idx = whole_chunk.columns.get_loc('Response')
        groups = whole_chunk.groupby('cc')
        for cc, group in groups:
            if cc in best_cc.index:
                compressed = group.iloc[:, maxisets.loc[cc]['unique_feature_list'] + [response_idx]]
                gdic[cc].append(compressed)
        i += 1
        print i
        if i == nchunks:
            break

    for cc, l in gdic.iteritems():
        full = pd.concat(l)
        with open('group_%i.pkl' % cc, 'w') as f:
            cPickle.dump(full, f, protocol=-1)
        print cc, pd.concat(l).count(axis=1).mean()


if __name__ == "__main__":
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
    # # p.print_callers('isinstance')
