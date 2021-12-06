import numpy as np
import os

from time import time
from sklearn.cluster import MiniBatchKMeans, KMeans
import json
import joblib
from classic_image_classification.utils.utils import check_n_make_dir


class VLAD:
    def __init__(self,
                 n_words=100,
                 cluster_method="MiniBatchKMeans",
                 normalize=False):

        self.cluster_method = cluster_method
        self.n_words = n_words
        self.parameters = dict()
        self.parameters["normalize"] = normalize

        self.k_means_clustering = None

    def _init_cluster_method(self, cluster_method):
        if cluster_method == "MiniBatchKMeans":
            k_means_clustering = MiniBatchKMeans(n_clusters=self.n_words,
                                                 init_size=2 * self.n_words)
        elif cluster_method == "KMeans":
            k_means_clustering = KMeans(n_clusters=self.n_words)
        else:
            raise ValueError("cluster method not recognised")
        return k_means_clustering

    def is_fitted(self):
        if self.k_means_clustering is not None:
            return True
        else:
            return False

    def save(self, model_path):
        check_n_make_dir(model_path)
        save_file_cluster = os.path.join(model_path, "vlad.pkl")
        save_file_parameters = os.path.join(model_path, "parameters.json")
        joblib.dump(self.k_means_clustering, save_file_cluster)

        with open(save_file_parameters, "w") as f:
            j_file = json.dumps(self.parameters)
            f.write(j_file)

    def load(self, model_path):
        save_file_cluster = os.path.join(model_path, "vlad.pkl")
        save_file_parameters = os.path.join(model_path, "parameters.json")
        if os.path.isfile(save_file_cluster):
            self.k_means_clustering = joblib.load(save_file_cluster)
            self.n_words = self.k_means_clustering.n_clusters
        else:
            self.k_means_clustering = None

        if os.path.isfile(save_file_parameters):
            with open(save_file_parameters) as json_file:
                self.parameters = json.load(json_file)

    def _remove_empty_desc(self, descriptors):
        """
        Empty descriptors can not be used to find visual words and are deleted before building visual word dictionary
        Args:
            descriptors:

        Returns:

        """
        descriptors_out = []
        for desc in descriptors:
            if desc is not None:
                descriptors_out.append(desc)
        return descriptors_out

    def fit(self, descriptors):
        self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = self._remove_empty_desc(descriptors)
        descriptors = np.concatenate(descriptors, axis=0)
        print("Fitting Visual Dictionary (n_words={}) to feature space...".format(self.n_words))
        print("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        print("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.fit(descriptors.astype("double"))
        print("done in %0.3fs" % (time() - t0))

    def partial_fit(self, descriptors):
        if self.k_means_clustering is None:
            self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = np.concatenate(descriptors, axis=0)
        print("Fitting VLAD to feature space...")
        print("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        print("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.partial_fit(descriptors)
        print("done in %0.3fs" % (time() - t0))

    def _translate_to_visual_words(self, vector):
        word = self.k_means_clustering.predict(vector)
        return word

    def transform(self, desc_sets):
        word_bags = []
        for descriptors in desc_sets:
            word_bag = self.transform_single(descriptors)
            word_bags.append(word_bag)
        return word_bags

    def transform_single(self, descriptors):
        X = np.array(descriptors)
        predictedLabels = self.k_means_clustering.predict(X)
        centers = self.k_means_clustering.cluster_centers_
        labels = self.k_means_clustering.labels_
        k = self.k_means_clustering.n_clusters

        m, d = X.shape
        V = np.zeros([k, d])
        # computing the differences

        # for all the clusters (visual words)
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == i) > 0:
                # add the diferences
                V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

        V = V.flatten()
        # power normalization, also called square-rooting normalization
        V = np.sign(V) * np.sqrt(np.abs(V))

        # L2 normalization

        V = V / np.sqrt(np.dot(V, V))
        return np.reshape(V, (1, -1))