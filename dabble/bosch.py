import numpy as np
import pandas as pd
import itertools
import scipy.weave


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
                int the_max = std::min(i_max, (int)all_last(j));
                if (the_max - the_min < min_intersect - 1) continue;
                int tot = 0;
                for (int k = the_min; k <= the_max; ++k) {
                    if (m(i, k) && m(j, k)) ++tot;
                    if (tot >= min_intersect) {
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


def nonnullcols(line):
    return tuple(line.notnull().nonzero()[0])


def main():
    nchunks = 20
    chunksize = 5000
    path_categorical = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_categorical.csv"
    path_numeric = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_numeric.csv"
    reader_categorical = pd.read_table(path_categorical, chunksize=chunksize, header=0, sep=',')
    reader_numeric = pd.read_table(path_numeric, chunksize=chunksize, header=0, sep=',')
    i = 0
    all_sets = set()
    for chunk_categorical, chunk_numeric in itertools.izip(reader_categorical, reader_numeric):
        chunk_categorical.set_index('Id', inplace=True)
        chunk_numeric.set_index('Id', inplace=True)
        assert chunk_categorical.index.equals(chunk_numeric.index)
        whole_chunk = pd.concat((chunk_categorical, chunk_numeric), axis=1)
        all_sets.update(whole_chunk.apply(nonnullcols, axis=1).unique())
        i += 1
        print i, len(all_sets)
        if i == nchunks:
            break
    all_sets = list(all_sets)
    all_lengths = np.array([len(s) for s in all_sets])
    all_first = np.array([s[0] for s in all_sets])
    all_last = np.array([s[-1] for s in all_sets])
    print np.mean(all_lengths), np.median(all_lengths)
    m = np.zeros((len(all_sets), np.amax([np.amax(s) for s in all_sets if len(s)]) + 1), dtype=np.uint8)
    for i, s in enumerate(all_sets):
        m[i, s] = 1
    J = jaccard_graph(m, all_lengths, all_first, all_last, 0.9)
    # J2 = quick_jaccard_graph(m, all_lengths, 0.9)
    # assert np.array_equal(J, J2)
    # adj = ((J > 0) & (J < 0.1)).astype(np.int)
    # L = np.diag(np.sum(adj, axis=1)) - adj
    # u, s, v = np.linalg.svd(L)
    # null_dim = np.sum(s <= 1e-9)
    print len(J)


if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    try:
        profiler.runctx('main()', globals(), locals())
    finally:
        profiler.dump_stats('prof')

    p = pstats.Stats("prof")
    p.strip_dirs().sort_stats('time').print_stats(30)
