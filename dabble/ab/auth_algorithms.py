# Import any required libraries or modules.
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def plot_xy_data(user, raw_data):
    shape = (600, 480)
    im = np.zeros(shape + (3, ), dtype=np.uint8)
    x = raw_data[:, 4::5].astype(np.float) / raw_data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y = raw_data[:, 5::5].astype(np.float) / raw_data[:, 0][:, None].astype(np.float)  # y relative to screen height
    assert np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
    im[np.round(shape[0] * y).astype(np.int), np.round(shape[1] * x).astype(np.int), :] = 255
    cv2.imshow(user, im)
    cv2.waitKey(0)


def pca_converter(data, n_dim):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mu) / std
    u, s, vt = np.linalg.svd(normalized_data)
    vt = vt[:n_dim]
    return lambda x: np.dot((x - mu) / std, vt.T)


def preprocess_data(data):
    keypress_dt = data[:, 8::10] - data[:, 3::10]  # duration of each keystroke
    key_to_key_dt = data[:, 13::10] - data[:, 3:-10:10]  # interval between keystrokes
    x = data[:, 4::5].astype(np.float) / data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y = data[:, 5::5].astype(np.float) / data[:, 0][:, None].astype(np.float)  # y relative to screen height
    assert np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
    other_info_idx = (np.arange(6, 8)[None, :] + np.arange(0, data.shape[1] - 4, 5)[:, None]).ravel()
    other_info = data[:, other_info_idx]
    collected_data = np.hstack((keypress_dt, key_to_key_dt, x, y, other_info))
    return collected_data


def simple_gaussian(user, training_dataset, template):
    user_raw = training_dataset[user]
    user_pca = template['pca'](preprocess_data(user_raw))

    # template will consist of mean and std dev of each feature in pca space
    mean_pca = np.mean(user_pca, axis=0)
    std_pca = np.std(user_pca, axis=0)
    ignoreable = np.sum(np.abs(user_pca - mean_pca) / std_pca > 2, axis=0) > 30
    return mean_pca, std_pca, ignoreable


def scikit_classifier(user, training_dataset, template):
    all_users = training_dataset.keys()
    others_raw = np.vstack([training_dataset[u] for u in all_users if u != user])
    others_pca = template['pca'](preprocess_data(others_raw))
    user_raw = training_dataset[user]
    user_pca = template['pca'](preprocess_data(user_raw))

    clf = KNeighborsClassifier(5)
    clf.fit(np.vstack((user_pca, others_pca)),
            np.hstack((np.zeros(len(user_pca)), np.ones(len(others_pca)))))
    return clf


def lda(user, training_dataset, template):
    all_users = training_dataset.keys()
    others_raw = np.vstack([training_dataset[u] for u in all_users if u != user])
    others_pca = template['pca'](preprocess_data(others_raw))
    user_raw = training_dataset[user]
    user_pca = template['pca'](preprocess_data(user_raw))

    others_mu = np.mean(others_pca, axis=0)
    user_mu = np.mean(user_pca, axis=0)
    others_sigma = np.cov(others_pca.T)
    user_sigma = np.cov(user_pca.T)
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
    pca = pca_converter(all_preprocessed, n_dim=32)
    template = {'pca': pca}

    template['mean_pca'], template['std_pca'], template['ignore'] = simple_gaussian(user, training_dataset, template)
    template['ld_vector'], template['ld_threshold'] = lda(user, training_dataset, template)
    template['clf'] = scikit_classifier(user, training_dataset, template)
    return template


# Implement authentication method here.  Feel free to write any helper classes or functions required.
# Return the authtication score and threshold above which you consider it being a correct user.
def authenticate(instance, user, templates, mode='lda'):
    t = templates[user]
    user_pca = t['pca'](preprocess_data(instance[None, :])[0])

    user_lda_proj = np.dot(t['ld_vector'], user_pca)
    lda_score, lda_thr = user_lda_proj - t['ld_threshold'], 0.

    clf_score, clf_thr = float(t['clf'].predict([user_pca]) == 0), 0.5

    ig = t['ignore']
    distance = np.sqrt(np.mean(((user_pca[~ig] - t['mean_pca'][~ig]) / t['std_pca'][~ig]) ** 2))
    simple_score, simple_thr = distance, 1.3

    if mode == 'lda':
        return lda_score, lda_thr
    elif mode == 'classifier':
        return clf_score, clf_thr
    elif mode == 'simple':
        return simple_score, simple_thr
    elif mode == 'combined':
        return np.mean([lda_score > lda_thr, clf_score > clf_thr, simple_score > simple_thr]), 0.5
    else:
        raise Exception("Unrecognized mode: %s", mode)