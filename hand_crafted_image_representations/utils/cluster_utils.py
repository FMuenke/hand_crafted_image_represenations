import numpy as np


def chose_samples(x, y, target):
    nb_samples = x.shape[0]
    nb_features = x.shape[1]
    x_out = []
    for idx in range(nb_samples):
        if y[idx] == target:
            chosen = np.zeros((1, nb_features))
            chosen[0, :] = x[idx, :]
            x_out.append(chosen)
    if len(x_out) == 0:
        return None
    elif len(x_out) == 1:
        return x_out[0]
    else:
        return np.concatenate(x_out, axis=0)


def calculate_cluster_center(x):
    cluster_center = np.mean(x, axis=0)
    return cluster_center


def get_cluster_centers(x, y):
    unique_y = np.unique(y)
    cluster_centers = []
    for y_target in unique_y:
        x_chosen = chose_samples(x, y, y_target)
        c_center = calculate_cluster_center(x_chosen)
        cluster_centers.append(c_center)
    return cluster_centers


def evaluate_cluster_distances(cluster_centers):
    number_of_clusters = len(cluster_centers)
    distances = dict()
    for idx1 in range(number_of_clusters):
        distances[idx1] = dict()
        for idx2 in range(number_of_clusters):
            if idx1 < idx2:
                c_center_1 = cluster_centers[idx1]
                c_center_2 = cluster_centers[idx2]
                distances[idx1][idx2] = np.linalg.norm(c_center_1 - c_center_2)
    return distances
