import numpy as np
import pandas as pd
import itertools
import scipy.weave
import cPickle


def quick_jaccard_graph(m, all_lengths, min_jaccard):
    m = m.astype(np.int)  # to avoid overflow in the dot product!
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

def quick_connected_components(all_sets, edges):
    adj = np.zeros((len(all_sets), len(all_sets)), dtype=np.int)
    adj[edges[::2], edges[1::2]] = 1
    adj[edges[1::2], edges[::2]] = 1
    L = np.diag(np.sum(adj, axis=1)) - adj
    u, s, v = np.linalg.svd(L)
    null_dim = np.sum(s <= 1e-9)
    return null_dim



def jaccard_graph(m, all_lengths, all_first, all_last, min_jaccard):
    jaccard_coeff = min_jaccard / (1 + min_jaccard)
    code = '''
        py::list edges;
        for (int i = 1; i < Nm[0]; ++i) {
            int i_min = all_first(i);
            int i_max = all_last(i);
            int length_i = all_lengths(i);
            for (int j = 0; j < i; ++j) {
                int length_j = all_lengths(j);
                int u = length_i + length_j;
                int min_intersect = (int)(u * jaccard_coeff) + 1;
                if (std::min(length_i, length_j) < min_intersect) continue;
                int the_min = std::max(i_min, (int)all_first(j));
                int the_max = std::min(i_max, (int)all_last(j)) + 1;  // not inclusive
                for (int k = the_min; k < the_max; ++k) {
                    if (the_max - k < min_intersect) break;
                    if (m(i, k) && m(j, k)) --min_intersect;
                    if (min_intersect <= 0) {
                        edges.append(j);
                        edges.append(i);
                        break;
                    }
                }
            }
        }
        return_val = edges;
    '''
    edges = scipy.weave.inline(
        code, ['m', 'all_lengths', 'all_first', 'all_last', 'jaccard_coeff'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return edges


def connected_components(n_vertices, edges):
    parent = np.arange(n_vertices, dtype=np.int)
    rank = np.zeros(n_vertices, dtype=np.int)
    code = '''
        for (int i=0; i < edges.size(); i+=2) {
            int u = edges[i];
            int v = edges[i + 1];
            int root_u = u;
            int root_v = v;
            while (parent(root_u) != root_u) root_u = parent(root_u);
            while (parent(root_v) != root_v) root_v = parent(root_v);
            if (root_u == root_v) continue;
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
        code, ['edges', 'parent', 'rank'],
        type_converters=scipy.weave.converters.blitz, compiler='gcc',
        extra_compile_args=["-O3"]
    )
    return parent


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
    hash_dict = {}
    for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
        chunk_categorical.set_index('Id', inplace=True)
        chunk_numeric.set_index('Id', inplace=True)
        chunk_numeric.drop(['Response'], axis=1, inplace=True)  # every record has response, do not use it for grouping
        # assert chunk_categorical.index.equals(chunk_numeric.index)
        whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1)
        feature_indices = whole_chunk.apply(nonnullcols, axis=1)
        hashes = feature_indices.apply(hash)
        all_hashes.append(hashes)
        hash_dict.update(dict(zip(hashes, feature_indices)))
        i += 1
        print i
        if i == nchunks:
            break
    hash_series = pd.concat(all_hashes)
    unique_hashes = hash_series.unique()
    unique_feature_sets = [hash_dict[h] for h in unique_hashes]
    idx_series = hash_series.map(pd.Series(np.arange(len(unique_feature_sets)), index=unique_hashes))
    all_features = whole_chunk.columns.tolist()
    with open(filename, 'w') as f:
        cPickle.dump((unique_feature_sets, idx_series, all_features), f, protocol=-1)
    return


def load_feature_sets(filename):
    with open(filename, 'r') as f:
        unique_feature_sets, idx_series, all_features = cPickle.load(f)
    return unique_feature_sets, idx_series, all_features


def main():
    unique_feature_sets, idx_series, all_features = load_feature_sets('sets.pkl')
    unique_feature_sets = unique_feature_sets[:10000]
    idx_series = idx_series[idx_series < 10000]
    n_sets = len(unique_feature_sets)
    all_lengths = np.array([len(s) for s in unique_feature_sets])
    all_first = np.array([(s[0] if len(s) else -1) for s in unique_feature_sets])
    all_last = np.array([(s[-1] if len(s) else -1) for s in unique_feature_sets])
    print np.mean(all_lengths), np.median(all_lengths)
    m = np.zeros((len(unique_feature_sets), np.amax(all_last) + 1), dtype=np.uint8)
    for i, s in enumerate(unique_feature_sets):
        m[i, s] = 1
    edges = jaccard_graph(m, all_lengths, all_first, all_last, 0.9)
    # edges_2 = quick_jaccard_graph(m, all_lengths, 0.9)
    # assert np.array_equal(edges, edges_2)

    cc = connected_components(n_sets, edges)
    num_components = len(np.unique(cc))
    # num_components_2 = quick_connected_components(all_sets, edges)
    # assert num_components_2 == num_components

    print n_sets, num_components


if __name__ == "__main__":
    # main()
    import cProfile, pstats
    profiler = cProfile.Profile()
    try:
        profiler.runctx('main()', globals(), locals())
    finally:
        profiler.dump_stats('prof')

    p = pstats.Stats("prof")
    p.strip_dirs().sort_stats('cumtime').print_stats(100)
    p.print_callers('isinstance')
