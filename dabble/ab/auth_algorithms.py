# Import any required libraries or modules.
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import sys


class MetaParams:
    n_lda_ensemble = 101
    lda_ensemble_feature_fraction = 0.4
    mode = 'lda_ensemble'


# The following is a hacky container for Statistics computed from the
# whole training set; we don't want to have to recompute them again at every call
# to build_template (it becomes slow for parameter searches with cross validation),
# so we preserve it here between calls. The proper place to
# do this would be in main.py, but we don't want to touch that.
Global = lambda: None
Global.ready = False


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
    normalized_data = (data - mu) / std
    u, s, vt = np.linalg.svd(normalized_data)
    cut_idx = np.argmin(np.abs(np.cumsum(s * s) / np.sum(s * s) - explained_variance))
    vt = vt[:cut_idx + 1]
    return (lambda x, mu=mu, std=std, vt=vt: np.dot((x - mu) / std, vt.T)),\
           np.diag(s[:cut_idx + 1] ** 2 / (len(data) - 1))


def preprocess_data(data):
    '''
    Turn raw data into an array of hand-picked features useful for classification
    :param data: n_samples x n_raw_features numpy array
    :return: n_samples x n_processed_features array
    '''
    keypress_dt = data[:, 8::10] - data[:, 3::10]  # duration of each keystroke
    key_to_key_dt = data[:, 13::10] - data[:, 3:-10:10]  # interval between keystrokes
    x_down = data[:, 4::10].astype(np.float) / data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y_down = data[:, 5::10].astype(np.float) / data[:, 0][:, None].astype(np.float)  # y relative to screen height
    x_up = data[:, 9::10].astype(np.float) / data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y_up = data[:, 10::10].astype(np.float) / data[:, 0][:, None].astype(np.float)  # y relative to screen height
    size_down = data[:, 6::10]
    size_up = data[:, 11::10]
    pressure_down = data[:, 7::10]
    pressure_up = data[:, 12::10]

    assert np.all((x_down >= 0) & (x_down <= 1) & (y_down >= 0) & (y_down <= 1))
    assert np.all((x_up >= 0) & (x_up <= 1) & (y_up >= 0) & (y_up <= 1))
    touch_d = np.hypot(x_down - x_up, y_down - y_up)
    collected_data = np.hstack((keypress_dt, key_to_key_dt,
                                np.diff(x_down, axis=1), np.diff(y_down, axis=1),
                                touch_d,
                                size_down, size_up, pressure_down, pressure_up,
                                ))
    return collected_data


def get_random_feature_selector(n_all_features, feature_fraction, seed):
    '''
    Return a selector of random features from a data array
    :param n_all_features: total number of features
    :param feature_fraction: desired fraction of selected features
    :param seed: random seed for repeatable experiments
    :return: a function taking in full data and returning only the random features from it
    '''
    n_features = int(np.round(feature_fraction * n_all_features))
    rng = np.random.RandomState(seed)
    p = rng.permutation(n_all_features)[:n_features]
    return lambda x, p=p: x[..., p]


def simple_gaussian(user_pca):
    # template will consist of mean and std dev of each feature in pca space
    mean_pca = np.mean(user_pca, axis=0)
    std_pca = np.std(user_pca, axis=0)
    return mean_pca, std_pca


def scikit_classifier(user, training_dataset, generator=lambda:KNeighborsClassifier(5)):
    '''
    Train a given classifier on user vs others
    :param generator: a function creating a scikit classifier with fit and predict functions
    :return: the trained classifier
    '''
    all_users = training_dataset.keys()
    others_raw = np.vstack([training_dataset[u] for u in all_users if u != user])
    others_pca = Global.pca(preprocess_data(others_raw))
    user_raw = training_dataset[user]
    user_pca = Global.pca(preprocess_data(user_raw))

    clf = generator()
    clf.fit(np.vstack((user_pca, others_pca)),
            np.hstack((np.zeros(len(user_pca)), np.ones(len(others_pca)))))
    return clf


def lda(user_pca, all_pca_cov, n_all):
    '''
    Compute the Fisher discriminant vector and threshold to classify user vs others.
    :param user_pca: n_samples x n_pca_features array of user instances
    :param all_pca_cov: covariance matrix of the complete dataset; it is assumed that
        the user data was part of the dataset, and that the mean of the whole dataset
        is 0 for every feature
    :param n_all: number of samples that formed the complete dataset
    :return: Fisher discriminant vector, threshold
    '''
    n_user = len(user_pca)
    assert n_user < n_all - 1  # make sure the complete dataset has more than just the current user

    # We compute mean and variance for the user data directly, and infer the mean
    # and variance of the rest of the dataset from the covariance of the complete set
    # (and its mean, which is assumed zero)
    user_mu = np.mean(user_pca, axis=0)
    others_mu = - n_user * user_mu / (n_all - n_user)
    user_sigma = np.cov(user_pca.T)
    def sq_(x):
        return x[:, None] * x[None, :]
    others_sigma = ((n_all - 1) * all_pca_cov - (n_user - 1) * user_sigma\
                   - n_user * sq_(user_mu) - (n_all - n_user) * sq_(others_mu)) / (n_all - n_user - 1)

    ld_vector = np.dot(np.linalg.inv(user_sigma + others_sigma), user_mu - others_mu)  # order determines sign of criterion
    ld_vector /= np.linalg.norm(ld_vector)

    # find the threshold for equal false positives and false negatives
    user_proj_mu = np.dot(user_mu, ld_vector)
    others_proj_mu = np.dot(others_mu, ld_vector)
    user_proj_std = np.sqrt(np.dot(ld_vector, np.dot(user_sigma, ld_vector)))
    others_proj_std = np.sqrt(np.dot(ld_vector, np.dot(others_sigma, ld_vector)))
    ld_threshold = (others_proj_std * user_proj_mu + user_proj_std * others_proj_mu) / (user_proj_std + others_proj_std)
    return ld_vector, ld_threshold


def compute_feature_discriminabilities(each_preprocessed):
    '''
    Return a vector of discriminability for each feature
    :param each_preprocessed: list with one n_samples x n_features data matrix for each user
    :return: vector of discriminabilities (sqrt of the square of the difference of means divided by
        the sum of variances) for each feature
    '''
    n_users = len(each_preprocessed)
    each_mu = np.array([np.mean(m, axis=0) for m in each_preprocessed])  # n_users x n_features
    each_var = np.array([np.var(m, axis=0) for m in each_preprocessed])  # n_users x n_features
    # compute discriminability for each feature and pair of users
    pairwise_discriminability = (each_mu[:, None, :] - each_mu[None :, :]) ** 2 / (1e-6 + each_var[:, None, :] + each_var[None :, :])
    # compute discriminability of each feature as the average over pairs of users
    return np.sqrt(np.sum(pairwise_discriminability, axis=(0, 1)) / (n_users * (n_users - 1)))


def _prepare_global(training_dataset):
    '''
    Processing of the complete dataset, to be reused for each user
        - feature preprocessing
        - pca converter
        - selection of features and computation of covariances for ensemble lda
    :param training_dataset: the complete dataset
    :return: None. The Global container is initialized with all necessary data
    '''
    each_preprocessed = [preprocess_data(training_dataset[u]) for u in training_dataset]
    Global.feature_discriminabilities = compute_feature_discriminabilities(each_preprocessed)

    all_preprocessed = np.vstack(each_preprocessed)
    Global.n_all = len(all_preprocessed)
    Global.pca, Global.all_pca_cov = pca_converter(all_preprocessed, Global.feature_discriminabilities, explained_variance=0.98)
    if MetaParams.mode == 'lda_ensemble':
        Global.lda_ensemble = []
        for i in range(MetaParams.n_lda_ensemble):
            seed = np.random.randint(200000)
            feature_selector = get_random_feature_selector(all_preprocessed.shape[1],
                                                           feature_fraction=MetaParams.lda_ensemble_feature_fraction, seed=seed)
            selected_pca, selected_pca_cov = pca_converter(feature_selector(all_preprocessed),
                                                           feature_selector(Global.feature_discriminabilities),
                                                           explained_variance=0.99)
            Global.lda_ensemble.append({'selector': feature_selector, 'pca': selected_pca, 'pca_cov': selected_pca_cov})
    Global.ready = True


# Implement template building here.  Feel free to write any helper classes or functions required.
# Return the generated template for that user.
def build_template(user, training_dataset):
    if not Global.ready:
        _prepare_global(training_dataset)

    user_raw = training_dataset[user]
    user_preprocessed = preprocess_data(user_raw)

    template = {}
    if MetaParams.mode in ['lda', 'simple', 'combined']:
        user_pca = Global.pca(user_preprocessed)
        template['mean_pca'], template['std_pca'] = simple_gaussian(user_pca)
        template['ld_vector'], template['ld_threshold'] =\
            lda(user_pca, all_pca_cov=Global.all_pca_cov, n_all=Global.n_all)

    if MetaParams.mode == 'lda_ensemble':
        lda_ensemble = []
        for lda_item in Global.lda_ensemble:
            user_selected_pca = lda_item['pca'](lda_item['selector'](user_preprocessed))
            ld_vector, ld_threshold = lda(user_selected_pca, n_all=Global.n_all, all_pca_cov=lda_item['pca_cov'])
            lda_ensemble.append({'ld_vector': ld_vector, 'ld_threshold': ld_threshold})
        template['lda_ensemble'] = lda_ensemble

    if MetaParams.mode in ['nonlinear', 'combined']:
        template['clf_1'] = scikit_classifier(user, training_dataset, generator=lambda: KNeighborsClassifier(5))
        template['clf_2'] = scikit_classifier(user, training_dataset, generator=lambda: svm.LinearSVC(C=0.05, class_weight='balanced'))
    return template


# Implement authentication method here.  Feel free to write any helper classes or functions required.
# Return the authenttication score and threshold above which you consider it being a correct user.
def authenticate(instance, user, templates):
    mode = MetaParams.mode
    assert mode in ['lda', 'combined', 'lda_ensemble', 'nonlinear', 'simple'], ("Unrecognized mode: %s" % mode)
    t = templates[user]
    batch_mode = instance.ndim > 1
    if not batch_mode:
        instance = instance[None, :]
    preprocessed_instance = preprocess_data(instance)

    if mode in ['lda', 'combined']:
        user_pca = Global.pca(preprocessed_instance)
        user_lda_proj = np.dot(user_pca, t['ld_vector'])
        lda_score, lda_thr = user_lda_proj - t['ld_threshold'], np.zeros(len(user_lda_proj))

    if mode in ['nonlinear', 'combined']:
        user_pca = Global.pca(preprocessed_instance)
        clf_score_1, clf_thr_1 = (t['clf_1'].predict(user_pca) == 0).astype(np.float), 0.5 * np.ones(len(user_pca))
        clf_score_2, clf_thr_2 = (t['clf_2'].predict(user_pca) == 0).astype(np.float), 0.5 * np.ones(len(user_pca))

    if mode == 'simple':
        user_pca = Global.pca(preprocessed_instance)
        z = (user_pca - t['mean_pca']) / t['std_pca']
        distance = np.mean(np.abs(z) ** 2, axis=1) ** 0.5
        score, thr = distance, 1.2 * np.ones(len(distance))

    if mode == 'lda_ensemble':
        ensemble_scores = np.empty((len(preprocessed_instance), len(t['lda_ensemble'])))
        for i, sub_t in enumerate(t['lda_ensemble']):
            g_item = Global.lda_ensemble[i]
            user_selected_pca = g_item['pca'](g_item['selector'](preprocessed_instance))
            user_thinned_lda_proj = np.dot(user_selected_pca, sub_t['ld_vector'])
            ensemble_scores[:, i] = user_thinned_lda_proj - sub_t['ld_threshold']

        score = np.mean(ensemble_scores > 0, axis=1)
        thr = 0.5 * np.ones(len(score))

    if mode == 'lda':
        score, thr = lda_score, lda_thr
    elif mode == 'nonlinear':
        score, thr = clf_score_1, clf_thr_1
    elif mode == 'combined':
        score = np.mean(np.vstack((lda_score > lda_thr, clf_score_1 > clf_thr_1, clf_score_2 > clf_thr_2)), axis=0)
        thr = 0.5 * np.ones(len(score))

    if not batch_mode:
        assert score.shape == (1, )
        assert thr.shape == (1, )
        score, thr = score[0], thr[0]

    return score, thr


def cross_validate(full_dataset, print_results=False):
    '''
    n-fold cross-validation of given dataset
    :param full_dataset: dictionary of raw data for each user
    :param print_results: if True, print progress messages and results
    :return: (percentage of false rejects, percentage of false accepts)
    '''
    n_folds = 5  # for cross-validation
    all_false_accept = 0
    all_false_reject = 0
    all_true_accept = 0
    all_true_reject = 0
    for i in range(n_folds):
        # split full dataset into training and validation
        training_dataset = dict()
        validation_dataset = dict()
        for u in full_dataset.keys():
            n = len(full_dataset[u])
            idx = np.round(float(n) / n_folds * np.arange(n_folds + 1)).astype(np.int)
            n_validation = np.diff(idx)
            rolled_set = np.roll(full_dataset[u], -idx[i], axis=0)
            training_dataset[u] = rolled_set[n_validation[i]:, :]
            validation_dataset[u] = rolled_set[:n_validation[i], :]

        # reset global data
        Global.ready = False
        templates = {u: build_template(u, training_dataset) for u in training_dataset}

        # For each user test authentication.
        true_accept = 0
        false_reject = 0
        true_reject = 0
        false_accept = 0
        for u in training_dataset:
            # Test false rejections.
            (score, threshold) = authenticate(validation_dataset[u], u, templates)
            true_accept += np.sum(score > threshold)
            false_reject += np.sum(score <= threshold)

            # Test false acceptance.
            for u_attacker in validation_dataset:
                if u == u_attacker:
                    continue
                (score, threshold) = authenticate(validation_dataset[u_attacker], u, templates)
                false_accept += np.sum(score > threshold)
                true_reject += np.sum(score <= threshold)

        if print_results:
            print "fold %i: false reject rate: %.1f%%, false accept rate: %.1f%%" %\
                  (i, 100. * float(false_reject) / (false_reject + true_accept),
                   100. * float(false_accept) / (false_accept + true_reject))
        all_false_accept += false_accept
        all_false_reject += false_reject
        all_true_accept += true_accept
        all_true_reject += true_reject

    false_reject_percent = 100. * float(all_false_reject) / (all_false_reject + all_true_accept)
    false_accept_percent = 100. * float(all_false_accept) / (all_false_accept + all_true_reject)

    if print_results:
        print "Total: false reject rate: %.1f%%, false accept rate: %.1f%%" % (false_reject_percent, false_accept_percent)

    return false_reject_percent, false_accept_percent



if __name__ == "__main__":
    # Reading the data into the training dataset separated by user.
    data_training_file = open(sys.argv[1], 'rb')
    csv_training_reader = csv.reader(data_training_file, delimiter=',', quotechar='"')
    csv_training_reader.next()
    full_dataset = dict()
    for row in csv_training_reader:
        if row[0] not in full_dataset:
            full_dataset[row[0]] = np.array([]).reshape((0, len(row[1:])))
        full_dataset[row[0]] = np.vstack([full_dataset[row[0]], np.array(row[1:]).astype(float)])

    for feature_fraction in [0.4]:
        for n_lda_ensemble in [51]:
            n_trials = 10
            tot_rej = 0
            tot_acc = 0
            for _ in range(n_trials):
                MetaParams.feature_fraction = feature_fraction
                MetaParams.n_lda_ensemble = n_lda_ensemble
                rej, acc = cross_validate(full_dataset)
                tot_rej += rej
                tot_acc += acc
            print "feature fraction=%.2f, ensemble size=%i, false_rej=%.2f%%, false_acc=%.2f%%" % (feature_fraction, n_lda_ensemble, tot_rej / n_trials, tot_acc / n_trials)
