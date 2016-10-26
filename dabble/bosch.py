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
    nchunks = 300
    chunksize = 5000
    _, train_set_idx, _ = load_feature_sets('sets.pkl')
    all_responses = train_set_idx['Response']
    for split in ['train']:
        grouped = False
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_original.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        n_classes = 4
        pi = np.zeros(n_classes + 1)
        ni = pi.copy()
        station_variant = None
        for chunk in reader:
            chunk = chunk.set_index('Id')
            if not grouped:
                cols = chunk.columns
                st = cols.str.rsplit('_', n=1).str[0]
                groups = chunk.columns.groupby(st)
                grouped = True
            tchunk = chunk.T
            stations = tchunk.groupby(st)
            if station_variant is None:
                station_variant = (stations.std().fillna(0) > 1e-6).sum(axis=1)
            else:
                station_variant += (stations.std().fillna(0) > 1e-6).sum(axis=1)
            # count = stations.count()
            # count_summean = count.mean(axis=1)
            # count_std = count.std(axis=1)

            variant = []
            for station, columns in groups.iteritems():
                variant.append(chunk.loc[:, columns].std(axis=1).fillna(0) > 1e-6)
            tot_variant = pd.concat(variant, axis=1).sum(axis=1)
            responses = all_responses[tot_variant.index]
            for j in range(n_classes):
                pi[j] += ((tot_variant == j) & (responses>0.5)).sum()
                ni[j] += ((tot_variant == j) & (responses<0.5)).sum()
            pi[n_classes] += ((tot_variant >= n_classes) & (responses>0.5)).sum()
            ni[n_classes] += ((tot_variant >= n_classes) & (responses<0.5)).sum()
            total = int(np.round(np.sum(pi) + np.sum(ni)))
            percent_variants = 100. * (pi + ni) / total
            percent_responses = 100. * pi / (pi + ni)
            print "Total %i variant percents: %s; positive percents %s" %\
                  (total, ', '.join(['%.3f' % kk for kk in percent_variants]), ', '.join(['%.2f' % kk for kk in percent_responses]))
            if len(chunk) == chunksize:
                assert total == (i + 1) * chunksize
            if (i % 20) == 19:
                print station_variant
            i += 1
            if i == nchunks:
                break
        print station_variant


def examine_date_station(station_name):
    nchunks = 300
    chunksize = 5000
    _, train_set_idx, _ = load_feature_sets('sets.pkl')
    all_responses = train_set_idx['Response']
    for split in ['train']:
        grouped = False
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_original.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        all_stats = None
        for chunk in reader:
            chunk = chunk.set_index('Id')
            if not grouped:
                columns = chunk.columns.str.startswith(station_name)
                grouped = True
            stchunk = chunk.loc[:, columns]
            new_frame = pd.DataFrame({'dt': stchunk.max(axis=1) - stchunk.min(axis=1),
                                      'n_present': stchunk.count(axis=1), 'response': all_responses[stchunk.index]}, index=stchunk.index)
            if all_stats is None:
                all_stats = new_frame
            else:
                all_stats = all_stats.append(new_frame)
            if i % 20 == 19:
                print ("Station %s (%i features):" % (station_name, stchunk.shape[1]))
                print all_stats.describe()
            i += 1
            if i == nchunks:
                break
        print ("Final station %s (%i features):" % (station_name, stchunk.shape[1]))
        print all_stats.describe()


def clean_date_per_station():
    '''
    Summarize all date features in one value per station, except for stations L1_S24 and L1_S25 which have variations
    in their many features
    '''
    nchunks = 300
    chunksize = 5000
    variant = ['L1_S24', 'L1_S25']
    for split in ['train', 'test']:
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_original.csv"
        write_path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_lumped_by_station.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        grouped = False
        for chunk in reader:
            chunk = chunk.set_index('Id')
            exists = os.path.exists(write_path)
            if not grouped:
                cols = chunk.columns
                st = cols.str.rsplit('_', n=1).str[0]
                groups = chunk.columns.groupby(st)
                grouped = True
            simplified = []
            for station, columns in groups.iteritems():
                if station in variant:
                    simplified.append(chunk.loc[:, columns])
                else:
                    simplified.append(chunk.loc[:, columns].mean(axis=1).rename(station))
            tot_simplified = pd.concat(simplified, axis=1)
            assert chunk.index.equals(tot_simplified.index)
            if not exists:
                print "from %i to %i cols" % (chunk.shape[1], tot_simplified.shape[1])
            with open(write_path, 'a') as f:
                tot_simplified.to_csv(f, sep=',', header=not exists, na_rep='', float_format='%.2f')
            i += 1
            print i
            if i == nchunks:
                break

def clean_date_by_hash():
    '''
    After lumping dates by station (all but stations 24 and 25 summarized in one feature) and looking for unique
        hashes to eliminate duplicates (mainly in stations 24 and 25, but maybe also others) here is the list of
        unique date features.
    We process the already lumped-by-station files to keep only these features
    :return:
    '''
    uniques = ['L0_S0', 'L0_S1', 'L0_S10', 'L0_S11', 'L0_S12', 'L0_S13', 'L0_S14', 'L0_S15', 'L0_S16', 'L0_S17', 'L0_S18', 'L0_S19',
               'L0_S2', 'L0_S20', 'L0_S21', 'L0_S22', 'L0_S23', 'L0_S3', 'L0_S4', 'L0_S5', 'L0_S6', 'L0_S7', 'L0_S8', 'L0_S9',
               'L1_S24_D1018', 'L1_S24_D1062', 'L1_S24_D1116', 'L1_S24_D1135', 'L1_S24_D1155', 'L1_S24_D1158', 'L1_S24_D1163',
               'L1_S24_D1168', 'L1_S24_D1171', 'L1_S24_D1178', 'L1_S24_D1186', 'L1_S24_D1277', 'L1_S24_D1368', 'L1_S24_D1413',
               'L1_S24_D1457', 'L1_S24_D1511', 'L1_S24_D1522', 'L1_S24_D1536', 'L1_S24_D1558', 'L1_S24_D1562', 'L1_S24_D1566',
               'L1_S24_D1568', 'L1_S24_D1570', 'L1_S24_D1576', 'L1_S24_D1583', 'L1_S24_D1674', 'L1_S24_D1765', 'L1_S24_D1770',
               'L1_S24_D1809', 'L1_S24_D1826', 'L1_S24_D677', 'L1_S24_D697', 'L1_S24_D702', 'L1_S24_D772', 'L1_S24_D801',
               'L1_S24_D804', 'L1_S24_D807', 'L1_S24_D813', 'L1_S24_D818', 'L1_S24_D909', 'L1_S24_D999', 'L1_S25_D1854',
               'L1_S25_D1867', 'L1_S25_D1883', 'L1_S25_D1887', 'L1_S25_D1891', 'L1_S25_D1898', 'L1_S25_D1902', 'L1_S25_D1980',
               'L1_S25_D2058', 'L1_S25_D2098', 'L1_S25_D2138', 'L1_S25_D2180', 'L1_S25_D2206', 'L1_S25_D2230', 'L1_S25_D2238',
               'L1_S25_D2240', 'L1_S25_D2242', 'L1_S25_D2248', 'L1_S25_D2251', 'L1_S25_D2329', 'L1_S25_D2406', 'L1_S25_D2430',
               'L1_S25_D2445', 'L1_S25_D2471', 'L1_S25_D2497', 'L1_S25_D2505', 'L1_S25_D2507', 'L1_S25_D2509', 'L1_S25_D2515',
               'L1_S25_D2518', 'L1_S25_D2596', 'L1_S25_D2674', 'L1_S25_D2713', 'L1_S25_D2728', 'L1_S25_D2754', 'L1_S25_D2780',
               'L1_S25_D2788', 'L1_S25_D2790', 'L1_S25_D2792', 'L1_S25_D2798', 'L1_S25_D2801', 'L1_S25_D2879', 'L1_S25_D2957',
               'L1_S25_D2996', 'L1_S25_D3011', 'L2_S26', 'L2_S27', 'L2_S28', 'L3_S29', 'L3_S30', 'L3_S31', 'L3_S32', 'L3_S33',
               'L3_S34', 'L3_S35', 'L3_S36', 'L3_S37', 'L3_S38', 'L3_S39', 'L3_S40', 'L3_S41', 'L3_S42', 'L3_S43', 'L3_S44',
               'L3_S45', 'L3_S46', 'L3_S47', 'L3_S48', 'L3_S49', 'L3_S50', 'L3_S51']


    nchunks = 300
    chunksize = 5000
    for split in ['train', 'test']:
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_lumped_by_station.csv"
        write_path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date_clean.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        for chunk in reader:
            chunk = chunk.set_index('Id').loc[:, uniques]
            exists = os.path.exists(write_path)
            with open(write_path, 'a') as f:
                chunk.to_csv(f, sep=',', header=not exists, na_rep='', float_format='%.2f')
            i += 1
            print i
            if i == nchunks:
                break


def hash_date():
    '''
    Summarize all date features in one value per station, except for stations L1_S24 and L1_S25 which have variations
    in their many features
    '''
    nchunks = 300
    chunksize = 5000
    for split in ['train', 'test']:
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_date.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        hashed = []
        for chunk in reader:
            hashed.append(chunk.apply(lambda x: hash(tuple(x)), axis=0))
            i += 1
            print i
            if i == nchunks:
                break
        all_hashed = pd.concat(hashed, axis=1)
        all_all_hashes = all_hashed.apply(lambda x: hash(tuple(x)), axis=1)
        unique, idx = np.unique(all_all_hashes.values, return_index=True)
        print "From %i to %i unique" % (len(all_hashed), len(unique))
        print "Repeated features:", sorted(set(all_all_hashes.index.values).difference(all_all_hashes.index[idx].values))
        print "Unique features:"
        pd.options.display.max_rows = 9999
        print list(all_all_hashes.index[idx].values)
        print "Hashes:"
        print all_all_hashes


def clean_numeric_by_hash():
    '''
    Remove repeated numeric features
    '''
    repeated = ['L1_S25_F2190', 'L1_S25_F2481', 'L1_S25_F2764', 'L3_S29_F3367', 'L3_S29_F3370', 'L3_S29_F3385',
                'L3_S29_F3388', 'L3_S29_F3395', 'L3_S29_F3398', 'L3_S29_F3401', 'L3_S29_F3404', 'L3_S29_F3412',
                'L3_S29_F3421', 'L3_S29_F3424', 'L3_S29_F3439', 'L3_S29_F3442', 'L3_S29_F3449', 'L3_S29_F3452',
                'L3_S29_F3455', 'L3_S29_F3458', 'L3_S29_F3467', 'L3_S29_F3470', 'L3_S30_F3529', 'L3_S30_F3539',
                'L3_S30_F3549', 'L3_S30_F3559', 'L3_S30_F3614', 'L3_S30_F3619', 'L3_S30_F3659', 'L3_S30_F3694',
                'L3_S30_F3699', 'L3_S30_F3714', 'L3_S30_F3719', 'L3_S30_F3789', 'L3_S30_F3814', 'L3_S30_F3824',
                'L3_S33_F3861', 'L3_S33_F3863', 'L3_S36_F3922', 'L3_S36_F3924', 'L3_S47_F4168']

    nchunks = 300
    chunksize = 5000
    for split in ['train', 'test']:
        path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_numeric_original.csv"
        write_path = "/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/" + split + "_numeric_clean.csv"
        reader = pd.read_table(path, chunksize=chunksize, header=0, sep=',')
        i = 0
        for chunk in reader:
            chunk = chunk.set_index('Id').drop(repeated, axis=1)
            exists = os.path.exists(write_path)
            with open(write_path, 'a') as f:
                chunk.to_csv(f, sep=',', header=not exists, na_rep='', float_format='%.2f')
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
            // Centroid is the element with minimum sum of distances to all other elements
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


def make_magic_features():
    train_date = pd.read_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_date.csv",
                             header=0, sep=',').set_index('Id')
    test_date = pd.read_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/test_date.csv",
                            header=0, sep=',').set_index('Id')
    train_date['min_date'] = train_date.min(axis=1)
    test_date['min_date'] = test_date.min(axis=1)
    train_date['is_train'] = 1
    test_date['is_train'] = 0
    all_date = pd.concat([train_date[['min_date', 'is_train']], test_date[['min_date', 'is_train']]])
    sorted_date = all_date.sort_values(by='min_date').reset_index()
    sorted_date['magic_1'] = np.concatenate((sorted_date['Id'].values[1:] - sorted_date['Id'].values[:-1], [0]))
    sorted_date['magic_2'] = np.concatenate(([0], sorted_date['Id'].values[1:] - sorted_date['Id'].values[:-1]))
    sorted_train = sorted_date[sorted_date['is_train'] == 1]
    sorted_test = sorted_date[sorted_date['is_train'] == 0]
    magic_train = sorted_train.set_index('Id').sort_index()[['min_date', 'magic_1', 'magic_2']]
    magic_test = sorted_test.set_index('Id').sort_index()[['min_date', 'magic_1', 'magic_2']]
    magic_train.to_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_magic.csv",
                       sep=',', header=True, na_rep='')
    magic_test.to_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/test_magic.csv",
                       sep=',', header=True, na_rep='')


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


def cluster_maxisets(filename, n_clusters, exclude_files=None):
    '''
    :param filename:
    :param n_clusters:
    :param exclude_files: if not None, 2-tuple (train_excluded_file, test_excluded_file); they
        should each have at least one column headed 'Id' with the ids to ignore in training and testing,
        respectively
    '''
    unique_feature_sets, train_set_idx, test_set_idx = load_feature_sets('sets.pkl')
    n_sets = None
    if n_sets is not None:
        unique_feature_sets = unique_feature_sets[:n_sets]
        train_set_idx = train_set_idx[train_set_idx['feature_set_idx'] < n_sets]
        test_set_idx = test_set_idx[test_set_idx['feature_set_idx'] < n_sets]

    if exclude_files is not None:
        old_ntrain = len(train_set_idx)
        old_ntest = len(test_set_idx)
        excluded_train = pd.read_csv(exclude_files[0], header=0, sep=',', usecols=['Id'])
        train_set_idx = train_set_idx.drop(excluded_train['Id'])
        excluded_test = pd.read_csv(exclude_files[1], header=0, sep=',', usecols=['Id'])
        test_set_idx = test_set_idx.drop(excluded_test['Id'])
        print "Dropped from %i to %i train and %i to %i test records" % (old_ntrain, len(train_set_idx), old_ntest, len(test_set_idx))

    best = (-1, -1, -1)
    for _ in range(100):
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
        record_averaged_J = (maxisets['mean_j'] * test_record_counts).sum() / test_record_counts.sum()
        record_averaged_n_features = n_features_per_record.mean()
        record_averaged_cluster_size = (test_record_counts * test_record_counts).sum() / test_record_counts.sum()
        record_averaged_n_maxifeatures = (maxisets['feature_count'] * all_record_counts).sum() / all_record_counts.sum()
        record_averaged_entropy = (-((1 - response_rate) * np.log(1 - response_rate) +
                                     response_rate * np.log(response_rate + 1e-16)) * train_record_counts).sum() / train_record_counts.sum()
        tot_response_rate = train_set_idx['Response'].mean()
        tot_entropy = -(tot_response_rate * np.log(tot_response_rate) + (1 - tot_response_rate) * np.log(1 - tot_response_rate))
        entropy_gain = (tot_entropy - record_averaged_entropy) / tot_entropy * 100
        print("Test-record-averaged J: %.3f, record averaged test cl size: %.0f, Std/mean test record count: %.2f, "
              "record_averaged maxi / n features: %.2f, entropy gain: %.1f%%" %
              (record_averaged_J, record_averaged_cluster_size, test_record_counts.std() / test_record_counts.mean(),
               record_averaged_n_maxifeatures / record_averaged_n_features, entropy_gain))

        new_best = (record_averaged_J, all_record_counts.mean() / all_record_counts.std(), entropy_gain)
        if np.sum([n > b for n, b in zip(new_best, best)]) >= 2:
            print "This is a better one!"
            best = new_best
            winner = (maxisets, train_set_idx.copy(), test_set_idx.copy())

    with open(filename, 'w') as f:
        cPickle.dump(winner, f, protocol=-1)
    print "All sets:"
    pd.options.display.max_rows = 9999
    print winner[0][['all_record_count', 'train_record_count', 'set_count', 'max_j', 'min_j', 'mean_j', 'feature_count', 'response_rate']].sort_values(by=['all_record_count'])
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
    if key in ['response', 'magic']:
        return pd.DataFrame({
            'presence': data.count(axis=0) / float(len(data)),
            'mean': data.mean(axis=0)})
    raise Exception('Stats of what? Key %s not recognized' % (key, ))


def _mcc(truth, x):
    '''
    Matthews correlation coefficient
    '''
    TP = np.sum(truth * x)
    TN = np.sum((1 - truth) * (1 - x))
    FP = np.sum((1 - truth) * x)
    FN = np.sum(truth * (1 - x))
    assert np.abs(TP + FP + TN + FN - len(truth) < 1e-6)
    score = ((TP * TN) - (FP * FN)) / np.sqrt(1e-12 + (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return score, TP, TN, FP, FN


def best_mcc(truth, p):
    regularization_TP = 0.5
    regularization_TN = 0.5
    idx = np.argsort(p)
    ordered_truth = truth[idx]
    all_FN = np.concatenate(([0], np.cumsum(ordered_truth))).astype(np.float)
    all_TN = np.concatenate(([0], np.cumsum((1 - ordered_truth)))).astype(np.float)
    all_TP = all_FN[-1] - all_FN
    all_FP = all_TN[-1] - all_TN
    all_TN += regularization_TN
    all_TP += regularization_TP
    score = ((all_TP * all_TN) - (all_FP * all_FN)) / np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN)
                                                              * (all_TN + all_FP) * (all_TN + all_FN))
    best = np.argmax(score)

    ordered_p = np.concatenate(([0.], p[idx] + 1e-12 * np.arange(len(p)), [1.]))
    thr = ordered_p[:-1] + np.diff(ordered_p) / 2
    best_thr = thr[best]
    # And now remove regularization optimism
    real_best_score, TP, TN, FP, FN = _mcc(truth, p > best_thr)
    # inv_idx = np.argsort(idx)
    # dithered_p = (p[idx] + 1e-12 * np.arange(len(p)))[inv_idx]
    # for i, test_p in enumerate(thr):
    #     mcc, TP, TN, FP, FN = _mcc(truth, dithered_p > test_p)
    #     assert mcc == score[i]
    #     assert TP == all_TP[i]
    #     assert TN == all_TN[i]
    #     assert FP == all_FP[i]
    #     assert FN == all_FN[i]
    return real_best_score, best_thr, (TP, TN, FP, FN)


def mcc_eval(p, dtrain):
    truth = dtrain.get_label()
    m = best_mcc(truth, p)[0]
    return 'MCC', m


def get_xgb_params(light_mode):
    params = {}
    params['booster'] = "gbtree"
    params['rate_drop'] = 0.5  # only meaningful if booster is 'dart'
    params['lambda'] = 1.0  # L2 regularization; only meaningful if booster is 'gblinear'
    params['alpha'] = 1.0  # L1 regularization; only meaningful if booster is 'gblinear'
    params['objective'] = "binary:logistic"
    params['eta'] = 0.02 if light_mode else 0.002
    params['learning_rate'] = params['eta']  # docs are so bad I dont know if it is eta or learning_rate or either
    params['gamma'] = 1.
    params['max_depth'] = 15
    params['colsample_bytree'] = 0.7 if light_mode else 0.7
    params['colsample_bylevel'] = 0.7 if light_mode else 0.7
    params['min_child_weight'] = 0.1
    params['base_score'] = 0.005
    params['silent'] = True
    return params


def get_xgb_clf():
    params = get_xgb_params(light_mode=False)
    params['n_estimators'] = 500
    return xgboost.XGBClassifier(**params)


def _fill_nans(train_dict, test_dict, stats):
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

    for k in train_dict.keys():
        stats[k] = _make_stats(k, train_dict[k])
    # print "Filled nans:"
    # print stats['categorical']
    # print stats['date']
    # print stats['numeric']
    return train_dict, test_dict, stats


def _predict_cluster(train_dict, test_dict, other_positives_dict, light_mode,
                     remove_absent=True, remove_invariant=True):
    '''
    :param train_dict: with keys 'categorical', 'date', 'numeric', 'magic', and 'response', each a frame
    :param test_dict: with keys 'categorical', 'date', 'numeric' and 'magic', each a frame
    :param other_positives_dict: with keys 'categorical', 'date', 'numeric' and 'magic', each a frame
    '''
    fill_nans = False
    normalize_mean_and_std = False
    pick_only_negative_neighbors = False
    resample = False
    stats = {k: _make_stats(k, v) for k, v in train_dict.iteritems()}
    # print stats['categorical']
    # print stats['date']
    # print stats['numeric']
    # print stats['response']
    # print "categorical:", len(train_dict['categorical'].columns), "date:", len(train_dict['date'].columns), "numeric:", len(train_dict['numeric'].columns)
    if remove_absent:
        present_dict = {k: stats[k]['presence'] > thr for k, thr in
                        [('categorical', 0.05), ('date', 0.05), ('numeric', 0.05), ('magic', 0.5), ('response', 0.)]}
        train_dict = {k: train_dict[k].loc[:, present_dict[k]] for k in train_dict.keys()}
        test_dict = {k: test_dict[k].loc[:, present_dict[k]] for k in test_dict.keys()}
        other_positives_dict = {k: other_positives_dict[k].loc[:, present_dict[k]] for k in other_positives_dict.keys()}
        stats = {k: _make_stats(k, train_dict[k]) for k in train_dict.keys()}
        # print "Removed absents:"
        # print stats['categorical']
        # print stats['date']
        # print stats['numeric']
        # print stats['response']
        # print "After absents categorical:", len(train_dict['categorical'].columns), "date:", len(train_dict['date'].columns), "numeric:", len(train_dict['numeric'].columns)

    # Always remove numeric and date features that have no variation in the training set
    # Categorical gets special conditions
    train_dict['magic']['magic_1'] = 1 * (train_dict['magic']['magic_1'] > 1)
    train_dict['magic']['magic_2'] = 1 * (train_dict['magic']['magic_2'] > 1)
    variant_conditions = [('numeric', 'std', 0.), ('date', 'std', 0.)]
    if remove_invariant:
        variant_conditions.append(('categorical', 'unique', 1))
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
    # print "After invariants categorical:", len(train_dict['categorical'].columns), "date:", len(train_dict['date'].columns), "numeric:", len(train_dict['numeric'].columns)

    # Time to select meaningful positives and join them to the training set
    informations = [('categorical', 0.1), ('date', 0.5), ('numeric', 0.95)]
    informative_positives_by_type = [other_positives_dict[k].count(axis=1) > thr * stats[k]['presence'].sum()
                                     for k, thr in informations if stats[k]['presence'].sum() > 0]
    informative_positives = pd.concat(informative_positives_by_type, axis=1).all(axis=1)
    for k in ['categorical', 'date', 'numeric', 'magic']:
        other_positives_dict[k] = other_positives_dict[k][informative_positives]
        train_dict[k] = pd.concat((train_dict[k], other_positives_dict[k]))
    train_dict['response'] = pd.concat((train_dict['response'], pd.DataFrame({'Response':np.ones(len(other_positives_dict['numeric'])),
                                                                              'Id': other_positives_dict['numeric'].index.values}).set_index('Id')))

    train_dict['date'] = train_dict['date'].sub(train_dict['magic']['min_date'], axis='rows')
    test_dict['date'] = test_dict['date'].sub(test_dict['magic']['min_date'], axis='rows')
    train_dict['date']['weekday'] = train_dict['magic']['min_date'] % 16.8
    test_dict['date']['weekday'] = test_dict['magic']['min_date'] % 16.8

    if fill_nans:
        train_dict, test_dict, stats = _fill_nans(train_dict, test_dict, stats)

    all_train_data = pd.concat((train_dict[k] for k in ['categorical', 'date', 'numeric', 'magic']), axis=1)
    all_test_data = pd.concat((test_dict[k] for k in ['categorical', 'date', 'numeric', 'magic']), axis=1)

    if normalize_mean_and_std:
        all_stats = _make_stats('numeric', pd.concat((all_train_data, all_test_data), axis=0))
        all_train_data = (all_train_data - all_stats['mean']) / (1e-9 + all_stats['std'])
        all_test_data = (all_test_data - all_stats['mean']) / (1e-9 + all_stats['std'])

    if pick_only_negative_neighbors:
        positives = all_train_data[(train_dict['response'] == 1).values]
        negatives = all_train_data[(train_dict['response'] == 0).values]
        nn = NearestNeighbors(n_neighbors=40).fit(negatives)
        d, nearest = nn.kneighbors(positives)
        nearest = np.unique(nearest.ravel())
        all_train_data = pd.concat((positives, negatives.iloc[nearest.ravel()]))

    if (resample):
        r = train_dict['response'].values.ravel()
        positive_rate = np.mean(r)

        sample_weights = (1 - positive_rate) * r + positive_rate * (1 - r)
        train_resample = pd.concat((all_train_data, pd.DataFrame({'Response': r, 'Id':
             all_train_data.index.values}).set_index('Id')), axis=1).sample(frac=1.0, replace=True, weights=sample_weights)
        the_response = train_resample['Response']
        the_train_data = train_resample.drop('Response', axis=1)
    else:
        the_train_data = all_train_data
        the_response = train_dict['response']

    # print("Added %i other positives, resampled positives from %.1f%% to %.1f%%; total samples from %i to %i, %i features" %
    #       (len(other_positives_dict['numeric']), train_dict['response'].mean() * 100, the_response.mean() * 100,
    #        len(all_train_data), len(the_train_data), len(the_train_data.columns)))
    #
    the_test_data = all_test_data
    # clf = SVC(C=0.05, kernel='linear')
    # clf = LinearSVC(C=0.01, penalty='l1', dual=False, class_weight='balanced')
    # the_feats = ['L0_S10_F224', 'L0_S6_F122', 'L0_S2_F60'
    #              'L1_S24_F1846', 'L3_S32_F3850',
    #              'L1_S24_F1695', 'L1_S24_F1632',
    #              'L3_S33_F3855', 'L1_S24_F1604',
    #              'L3_S29_F3407', 'L3_S33_F3865', 'L3_S36_F3920', 'L3_S36_F3859',
    #              'L3_S38_F3952', 'L1_S24_F1723']
    # the_present_feats = list(set(the_train_data.columns).intersection(the_feats))
    # the_train_data = the_train_data[the_present_feats]
    # the_test_data = the_test_data[the_present_feats]
    # print "Reduced to %i cols" % len(the_train_data.columns)
    # clf = get_xgb_clf()

    # the_eval_data = xgboost.DMatrix(the_train_data.iloc[::3], the_response.iloc[::3], silent=True,
    #                                  feature_names=the_train_data.columns)
    # the_train_data = xgboost.DMatrix(pd.concat((the_train_data.iloc[1::3], the_train_data.iloc[2::3])),
    #                                  pd.concat((the_response.iloc[1::3], the_response.iloc[2::3])),
    #                                  silent=True, feature_names=the_train_data.columns)
    the_train_data = xgboost.DMatrix(the_train_data, the_response,
                                     silent=True, feature_names=the_train_data.columns)
    the_test_data = xgboost.DMatrix(the_test_data, silent=True,
                                    feature_names=the_test_data.columns)
    # evals_result = {}
    params = get_xgb_params(light_mode=light_mode)
    clf = xgboost.train(params, the_train_data,
                        # evals=[(the_eval_data, 'eval')], early_stopping_rounds=40, feval=mcc_eval, maximize=True, verbose_eval=False, evals_result=evals_result,
                        num_boost_round=50 if light_mode else 250)
    # ntree_limit = clf.best_iteration + 1
    # print "Best iteration:", ntree_limit, evals_result
    prob = clf.predict(the_test_data)#, ntree_limit=ntree_limit)

    # clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=12,
    #                              max_features=min(len(the_train_data.columns), int(4 * np.sqrt(len(the_train_data.columns)))),
    #                              class_weight='balanced')
    # clf.fit(the_train_data, the_response.values.ravel())
    # prob = clf.predict_proba(the_test_data)[:, 1]
    # the_train_data = pca(the_train_data)
    # the_test_data = pca(the_test_data)
    # clf = Convnet(data_length=the_data.shape[1], n_channels=1, n_outputs=2,
    #               kernel_defs=[(1, 4, 1)], fc_sizes=[16],
    #               batch_size=128, n_passes=60, dropout_keep_prob=0.5)
    # clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1., min_samples_leaf=5, max_depth=10)
    # clf2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'), n_estimators=50)
    # clf2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # clf2 = RadiusNeighborsClassifier()
    # fi = clf.feature_importances_
    # clf2.fit(the_train_data * fi, the_response)
    return prob


def  cross_eval(train_dict, other_positives_dict, light_mode, verbose):
    kf = KFold(len(train_dict['response']), n_folds=5)
    remove_absent = True
    remove_invariant = True
    pd.options.display.float_format = '{:,.2f}'.format
    all_prob = np.array([])
    all_results = np.array([])
    for train, test in kf:
        train_chunk_dict = {k: train_dict[k].iloc[train] for k in train_dict.keys()}
        validation_dict = {k: train_dict[k].iloc[test] for k in train_dict.keys()}
        prob = _predict_cluster(train_chunk_dict, validation_dict, other_positives_dict, light_mode=light_mode,
                                remove_absent=remove_absent, remove_invariant=remove_invariant)
        all_prob = np.append(all_prob, prob)
        all_results = np.append(all_results, validation_dict['response'].values.ravel())
        score, best_p, (TP, TN, FP, FN) = best_mcc(validation_dict['response'].values.ravel(), prob)
        if verbose:
            print "Score: %.3f, best_p: %.4f, positives %i, true %i, false %i" % (score, best_p, np.sum(validation_dict['response']), TP, FP)
    score, best_p, (TP, TN, FP, FN) = best_mcc(all_results, all_prob)
    return score, best_p, (TP, TN, FP, FN)


def append_results(filepath, frame, index):
    exists = os.path.exists(filepath)
    with open(filepath, 'a') as f:
        frame.to_csv(f, sep=',', header=not exists, na_rep='', index=index)


def main():
    dirname = 'clustereddata_300_clean'
    timestr = time.strftime('%y%m%d_%H%M%S', time.localtime())
    results_path = os.path.join(dirname, 'results_' + timestr + '.csv')
    exclude_path = os.path.join(dirname, 'excluded_' + timestr + '.csv')
    all_positives = {}
    with open('positives.pkl', 'r') as f:
        all_positives['categorical'], all_positives['date'], all_positives['numeric'] = cPickle.load(f)
    train_magic = pd.read_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/train_magic.csv",
                              header=0, sep=',').set_index('Id')
    test_magic = pd.read_csv("/media/psf/Home/linux-home/Borja/Cursos/kaggle/bosch/test_magic.csv",
                             header=0, sep=',').set_index('Id')
    all_positives['magic'] = train_magic.loc[all_positives['date'].index]

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
    for idx in all_idx[0:300]:
        cc = train_files['numeric'][idx][-10:-4]
        train_dict = {}
        test_dict = {}
        for k in ['categorical', 'date', 'numeric', 'response']:
            train_file = os.path.join(dirname, [s for s in train_files[k] if cc in s][0])
            train_dict[k] = pd.read_csv(train_file, sep=',', header=0).set_index('Id')
        for k in ['categorical', 'date', 'numeric']:
            test_file = os.path.join(dirname, [s for s in test_files[k] if cc in s][0])
            test_dict[k] = pd.read_csv(test_file, sep=',', header=0).set_index('Id')
        train_dict['magic'] = train_magic.loc[train_dict['date'].index]
        test_dict['magic'] = test_magic.loc[test_dict['date'].index]
        tot_records += len(train_dict['categorical'])
        print("Processing %s, size=%i, n_records=%i" % (cc, sizes[idx], len(train_dict['categorical'])))
        others = ~all_positives['categorical'].index.isin(train_dict['categorical'].index)
        other_positives_dict = {k: all_positives[k].loc[others, train_dict[k].columns] for k in all_positives.keys()}
        # print("Kept %i extra positives out of %i" % (len(other_positives['categorical']), len(all_positives['categorical'])))

        score, best_p, (TP, TN, FP, FN) = cross_eval(train_dict, other_positives_dict, light_mode=True, verbose=False)
        if score > 0.1:
            print "Going heavy here, got a score of %.3f..." % (score, )
            score, best_p, (TP, TN, FP, FN) = cross_eval(train_dict, other_positives_dict, light_mode=False, verbose=False)
            if score > 0.3:
                print "This is the real thing, score %.3f, best_p=%.3f, doing the test data..." % (score, best_p)
                prob = _predict_cluster(train_dict, test_dict, other_positives_dict, light_mode=False,
                                        remove_absent=True, remove_invariant=True)
                append_results(results_path, pd.DataFrame(data={'Response': (prob > best_p).astype(np.int)}, index=test_dict['numeric'].index), index=True)
                append_results(exclude_path, pd.Series(train_dict['numeric'].index), index=False)
                print "Saved test and excluded data for cluster %s!" % (cc, )

        all_TP += TP
        all_TN += TN
        all_FP += FP
        all_FN += FN
        print "Final cluster score: %.3f, best_p: %.4f, positives %i, true %i, false %i" % (score, best_p, np.sum(train_dict['response']), TP, FP)
        print "Cum records:", tot_records, "Cumulative score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
            np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))
        print

    print "Final score:", ((all_TP * all_TN) - (all_FP * all_FN)) /\
                          np.sqrt(1e-12 + (all_TP + all_FP) * (all_TP + all_FN) * (all_TN + all_FP) * (all_TN + all_FN))


if __name__ == "__main__":
    # make_magic_features()
    # save_all_positives('kk.pkl')
    # compute_feature_sets('new_sets.pkl')
    # cluster_maxisets('maxisets_200_round2.pkl', 300, exclude_files=(os.path.join('clustereddata_300_clean/excluded_161024_152759.csv'),
    #                                                                 os.path.join('clustereddata_300_clean/results_161024_152759.csv')))

    # distribute_data_in_clusters('maxisets_200_round2.pkl')

    # clean_categorical()
    # examine_date()
    # examine_date_station('L2')
    # clean_date()
    # hash_date()
    # clean_numeric_by_hash()
    # clean_date_by_hash()

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
