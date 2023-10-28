import numpy as np
import random
from sklearn.cluster import KMeans


def sample_by_representation(representations, tags, n_samples):
    cluster = KMeans(n_clusters=n_samples, n_init="auto")
    cluster.fit(representations)
    y = cluster.predict(representations)
    unique_values, counts = np.unique(y, return_counts=True)

    sampled_tags = []

    # Iterate through unique values and select a random index for each
    for value in unique_values:
        indices = np.where(y == value)[0]  # Get indices for a specific unique value
        random_index = np.random.choice(indices)  # Choose a random index
        sampled_tags.append(tags[random_index])
    return sampled_tags


def sample_randomly(tags, n_samples):
    return random.sample(tags, n_samples)


class DataSampling:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample_by_representation(self, representations, tags, y=None):
        assert len(tags) > self.n_samples, "Not enough tags to sample! {}".format(len(tags))
        if y is None:
            return sample_by_representation(representations, tags, self.n_samples)
        else:
            unique_values, counts = np.unique(y, return_counts=True)
            n_samples_per_class = int(self.n_samples / len(unique_values))
            sampled_tags = []
            for cls_i, n_class in zip(unique_values, counts):
                assert n_class > n_samples_per_class, "Not enough tags to sample in class {}! {}".format(cls_i, n_class)
                indices = np.where(y == cls_i)[0]
                representations_cls = representations[indices, :]
                tags_cls = [tags[i] for i in indices]
                sampled_tags += sample_by_representation(representations_cls, tags_cls, n_samples_per_class)
            return sampled_tags

    def sample_randomly(self, tags, y=None):
        assert len(tags) > self.n_samples, "Not enough tags to sample! {}".format(len(tags))
        if y is None:
            return sample_randomly(tags, self.n_samples)
        else:
            unique_values, counts = np.unique(y, return_counts=True)
            n_samples_per_class = int(self.n_samples / len(unique_values))
            sampled_tags = []
            for cls_i, n_class in zip(unique_values, counts):
                assert n_class > n_samples_per_class, "Not enough tags to sample in class {}! {}".format(cls_i, n_class)
                indices = np.where(y == cls_i)[0]
                tags_cls = [tags[i] for i in indices]
                sampled_tags += sample_randomly(tags_cls, n_samples_per_class)
            return sampled_tags
