# Import any required libraries or modules.
import numpy as np
import cv2


def plot_data(user, data):
    shape = (600, 480)
    im = np.zeros(shape + (3, ), dtype=np.uint8)
    x = data[:, 4::5].astype(np.float) / data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y = data[:, 5::5].astype(np.float) / data[:, 0][:, None].astype(np.float)  # y relative to screen height
    assert np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
    im[np.round(shape[0] * y).astype(np.int), np.round(shape[1] * x).astype(np.int), :] = 255
    cv2.imshow(user, im)
    cv2.waitKey(0)


def pca(data, n_dim):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mu) / std
    u, s, vt = np.linalg.svd(normalized_data)
    vt = vt[:n_dim]
    return mu, std, vt, np.dot(normalized_data, vt.T)


def preprocess_data(data):
    keypress_dt = data[:, 8::10] - data[:, 3::10]  # duration of each keystroke
    key_to_key_dt = data[:, 13::10] - data[:, 3:-10:10]  # interval between keys
    x = data[:, 4::5].astype(np.float) / data[:, 1][:, None].astype(np.float)  # x relative to screen width
    y = data[:, 5::5].astype(np.float) / data[:, 0][:, None].astype(np.float)  # y relative to screen height
    assert np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
    other_info_idx = (np.arange(6, 8)[None, :] + np.arange(0, data.shape[1] - 4, 5)[:, None]).ravel()
    other_info = data[:, other_info_idx]
#    collected_data = np.hstack((x, y, other_info))
    collected_data = np.hstack((keypress_dt, key_to_key_dt, x, y, other_info))
    return collected_data

def lda(user, training_dataset, all_means, all_stds):
    all_users = training_dataset.keys()
    others_data = np.vstack([training_dataset[u] for u in all_users if u != user])
    others_preprocessed_data = (preprocess_data(others_data) - all_means) / all_stds
    user_data = training_dataset[user]
    user_preprocessed_data = (preprocess_data(user_data) - all_means) / all_stds

    others_mu = np.mean(others_preprocessed_data, axis=0)
    user_mu = np.mean(user_preprocessed_data, axis=0)
    others_sigma = np.cov(others_preprocessed_data)
    user_sigma = np.cov(user_preprocessed_data)
    ld_vector = np.dot(np.linalg.inv(user_sigma + others_sigma), user_mu - others_mu)
    ld_vector /= np.linalg.norm(ld_vector)

    user_proj = np.dot(user_preprocessed_data, ld_vector)
    others_proj = np.dot(others_preprocessed_data, ld_vector)



# Implement template building here.  Feel free to write any helper classes or functions required.
# Return the generated template for that user.
def build_template(user, training_dataset):
    all_users = training_dataset.keys()
    all_data = np.vstack([training_dataset[u] for u in all_users])
    all_preprocessed_data = preprocess_data(all_data)
    all_means, all_stds, all_vt, _ = pca(all_preprocessed_data, n_dim=3)
    ld_vector, ld_threshold = lda(user, training_dataset, all_means, all_stds)

    mu = None
    for i, u in enumerate(all_users):
        t = _make_template(training_dataset[u], all_vt, all_means, all_stds)
        if mu is None:
            mu = np.empty((len(all_users), t['mean_pca'].shape[0]))
            sigma = np.empty_like(mu)
        mu[i] = t['mean_pca']
        sigma[i] = t['std_pca']
    discriminability = np.std(mu, axis=0) / (np.mean(sigma, axis=0) + 1e-12)

    template = _make_template(training_dataset[user], all_vt, all_means, all_stds)
    template['discriminability'] = discriminability
    return template


def _make_template(user_data, all_vt, all_means, all_stds):
    user_preprocessed_data = preprocess_data(user_data)
    user_preprocessed_data = (user_preprocessed_data - all_means) / all_stds
    mu_preprocessed, std_preprocessed, user_vt, user_pca_data = pca(user_preprocessed_data, 2)
    user_pca_data = np.dot(user_preprocessed_data, all_vt.T)

    # template will consist of mean and std dev of each feature in pca space
    mean_pca_data = np.mean(user_pca_data, axis=0)
    std_pca_data = np.std(user_pca_data, axis=0)
    ignoreable = np.sum(np.abs(user_pca_data - mean_pca_data) / std_pca_data > 2, axis=0) > 3
    # Return the template to be used in the authenticate function.
    template = {'mean_preprocessed': mu_preprocessed, 'std_preprocessed': std_preprocessed,
                'mean_pca': mean_pca_data, 'std_pca': std_pca_data, 'ignore': ignoreable,
                'user_vt': user_vt, 'all_vt': all_vt, 'all_means': all_means, 'all_stds': all_stds}
    return template

# Implement authentication method here.  Feel free to write any helper classes or functions required.
# Return the authtication score and threshold above which you consider it being a correct user.
def authenticate(instance, user, templates):
    t = templates[user]
    user_preprocessed_data = preprocess_data(instance[None, :])[0]
#    user_pca_data = np.dot(t['user_vt'], (user_preprocessed_data - t['mean_preprocessed']) / t['std_preprocessed'])
    user_pca_data = np.dot(t['all_vt'], (user_preprocessed_data - t['all_means']) / t['all_stds'])

    ig = t['ignore']
    distance = np.abs(user_pca_data[~ig] - t['mean_pca'][~ig]) / t['std_pca'][~ig]
    d = np.sqrt(np.sum((distance * t['discriminability'][~ig]) ** 2)) / np.sqrt(np.sum(t['discriminability'][~ig] ** 2))
    # Convert distance to a score.
    score = 1.0/(1.0+np.exp(d))

    # Return score and threshold.
    return score, 0.15
