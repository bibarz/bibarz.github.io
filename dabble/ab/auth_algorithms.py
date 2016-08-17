# Import any required libraries or modules.
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv


def plot_xy_data(user, raw_data):
    shape = (600, 480)
    im = np.zeros(shape + (3, ), dtype=np.uint8)
    x = raw_data[:, 4::5].astype(np.float) / raw_data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y = raw_data[:, 5::5].astype(np.float) / raw_data[:, 0][:, None].astype(np.float)  # y relative to screen height
    assert np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
    im[np.round(shape[0] * y).astype(np.int), np.round(shape[1] * x).astype(np.int), :] = 255
    cv2.imshow(user, im)
    cv2.waitKey(0)


def pca_converter(data, explained_variance):
    '''
    :param explained_variance: ratio of explained variance (between 0 and 1) that will determine how many components are kept
    :return: function transforming data into pca components, and covariance matrix
        of transformed data
    '''
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mu) / std
    u, s, vt = np.linalg.svd(normalized_data)
    cut_idx = np.argmin(np.abs(np.cumsum(s * s) / np.sum(s * s) - explained_variance))
    vt = vt[:cut_idx + 1]
    return lambda x: np.dot((x - mu) / std, vt.T), np.diag(s[:cut_idx + 1] ** 2 / (len(data) - 1))


def preprocess_data(data):
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
    x_av = np.array([0.773, 0.502, 0.505, 0.517, 0.237, 0.256, 0.776])
    y_av = np.array([0.687, 0.372, 0.367, 0.841, 0.679, 0.368, 0.858])
    d_down = np.hypot(x_down - x_av, y_down - y_av)
    d_up = np.hypot(x_up - x_av, y_up - y_av)
    touch_d = np.hypot(x_down - x_up, y_down - y_up)
    collected_data = np.hstack((keypress_dt, key_to_key_dt,
                                # np.hypot(x_down[:, 1:] - x_up[:, :-1], y_down[:, 1:] - y_up[:, :-1]),
                                x_up, y_up,
                                np.diff(x_down, axis=1), np.diff(y_down, axis=1),
                                # touch_d,
                                size_down, size_up, pressure_down, pressure_up,
                                ))
    return collected_data


def simple_gaussian(user_raw, template):
    user_pca = template['pca'](preprocess_data(user_raw))

    # template will consist of mean and std dev of each feature in pca space
    mean_pca = np.mean(user_pca, axis=0)
    std_pca = np.std(user_pca, axis=0)
    ignoreable = np.sum(np.abs(user_pca - mean_pca) / std_pca > 2, axis=0) > len(user_pca) * 0.1
    return mean_pca, std_pca, ignoreable


def scikit_classifier(user, training_dataset, template, generator=lambda:KNeighborsClassifier(5)):
    all_users = training_dataset.keys()
    others_raw = np.vstack([training_dataset[u] for u in all_users if u != user])
    others_pca = template['pca'](preprocess_data(others_raw))
    user_raw = training_dataset[user]
    user_pca = template['pca'](preprocess_data(user_raw))

    clf = generator()
    clf.fit(np.vstack((user_pca, others_pca)),
            np.hstack((np.zeros(len(user_pca)), np.ones(len(others_pca)))))
    return clf


def lda(user_raw, template):
    user_pca = template['pca'](preprocess_data(user_raw))
    n_user = len(user_pca)
    n_all = template['n_all']
    assert n_user < n_all - 1  # make sure the complete dataset has more than just the current user

    user_mu = np.mean(user_pca, axis=0)
    others_mu = - n_user * user_mu / (n_all - n_user)
    user_sigma = np.cov(user_pca.T)
    def sq_(x):
        return x[:, None] * x[None, :]
    others_sigma = ((n_all - 1) * template['all_pca_cov'] - (n_user - 1) * user_sigma\
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


# Implement template building here.  Feel free to write any helper classes or functions required.
# Return the generated template for that user.
def build_template(user, training_dataset):
    all_users = training_dataset.keys()
    all_raw = np.vstack([training_dataset[u] for u in all_users])
    all_preprocessed = preprocess_data(all_raw)
    pca, all_pca_cov = pca_converter(all_preprocessed, explained_variance=0.99)
    template = {'pca': pca, 'all_pca_cov': all_pca_cov, 'n_all': len(all_preprocessed)}

    template['mean_pca'], template['std_pca'], template['ignore'] = simple_gaussian(training_dataset[user], template)
    template['ld_vector'], template['ld_threshold'] = lda(training_dataset[user], template)
    template['clf_1'] = scikit_classifier(user, training_dataset, template, generator=lambda: KNeighborsClassifier(5))
    template['clf_2'] = scikit_classifier(user, training_dataset, template, generator=lambda: svm.LinearSVC(C=0.05, class_weight='balanced'))
    return template


# Implement authentication method here.  Feel free to write any helper classes or functions required.
# Return the authtication score and threshold above which you consider it being a correct user.
def authenticate(instance, user, templates, mode='combined'):
    t = templates[user]
    user_pca = t['pca'](preprocess_data(instance[None, :])[0])

    user_lda_proj = np.dot(t['ld_vector'], user_pca)
    lda_score, lda_thr = user_lda_proj - t['ld_threshold'], 0.

    clf_score_1, clf_thr_1 = float(t['clf_1'].predict([user_pca]) == 0), 0.5
    clf_score_2, clf_thr_2 = float(t['clf_2'].predict([user_pca]) == 0), 0.5

    ig = t['ignore']
    z = (user_pca[~ig] - t['mean_pca'][~ig]) / t['std_pca'][~ig]
    distance = np.mean(np.abs(z) ** 2) ** 0.5
    simple_score, simple_thr = distance, 1.35

    if mode == 'lda':
        return lda_score, lda_thr
    elif mode == 'classifier':
        return clf_score_1, clf_thr_1
    elif mode == 'simple':
        return simple_score, simple_thr
    elif mode == 'combined':
        return np.mean([lda_score > lda_thr, clf_score_1 > clf_thr_1, clf_score_2 > clf_thr_2]), 0.5
    else:
        raise Exception("Unrecognized mode: %s", mode)


def cross_validate(data_training_file_name, data_testing_file_name):

    # Reading the data into the training dataset separated by user.
    data_training_file = open(data_training_file_name, 'rb')
    csv_training_reader = csv.reader(data_training_file, delimiter=',', quotechar='"')
    csv_training_reader.next()

    full_dataset = dict()

    for row in csv_training_reader:
        if row[0] not in full_dataset:
            full_dataset[row[0]] = np.array([]).reshape((0, len(row[1:])))
        full_dataset[row[0]] = np.vstack([full_dataset[row[0]], np.array(row[1:]).astype(float)])

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

        templates = {u: build_template(u, training_dataset) for u in training_dataset}

        # For each user test authentication.
        true_accept = 0
        false_reject = 0
        true_reject = 0
        false_accept = 0
        for u in training_dataset:
            # Test false rejections.
            for instance in validation_dataset[u]:
                (score, threshold) = authenticate(instance, u, templates)
                # If score higher than the threshold, we accept the user as authentic.
                if score > threshold:
                    true_accept += 1
                else:
                    false_reject += 1

            # Test false acceptance.
            for u_attacker in validation_dataset:
                if u == u_attacker:
                    continue
                for instance in validation_dataset[u_attacker]:
                    (score, threshold) = authenticate(instance, u, templates)
                    # If score lower of equal to the threshold, we reject the user as an attacker.
                    if score <= threshold:
                        true_reject += 1
                    else:
                        false_accept += 1

        print "fold %i: bad reject rate: %.1f, bad accept rate: %.1f" % (i,
                                                                         100. * float(false_reject) / (false_reject + true_accept),
                                                                         100. * float(false_accept) / (false_accept + true_reject))
        all_false_accept += false_accept
        all_false_reject += false_reject
        all_true_accept += true_accept
        all_true_reject += true_reject

    print "Total: bad reject rate: %.1f, bad accept rate: %.1f" % (100. * float(all_false_reject) / (all_false_reject + all_true_accept),
                                                                   100. * float(all_false_accept) / (all_false_accept + all_true_reject))


if __name__ == "__main__":
    cross_validate('dataset_training.csv', 'dataset_testing.csv')
