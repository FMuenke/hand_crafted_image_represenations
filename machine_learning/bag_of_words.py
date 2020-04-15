import joblib
import json
import os
from time import time
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from utils.utils import check_n_make_dir


class BagOfWords:
    def __init__(self, model_path,
                 n_words=100,
                 cluster_method="MiniBatchKMeans",
                 normalize=False):

        self.n_words = n_words
        self.parameters = dict()
        self.parameters["normalize"] = normalize

        self.k_means_clustering = self._init_cluster_method(cluster_method)

        self.model_path = model_path
        self.save_file_cluster = os.path.join(self.model_path, "bag_of_words.pkl")
        self.save_file_parameters = os.path.join(self.model_path, "parameters.json")

    def _init_cluster_method(self, cluster_method):
        if cluster_method == "MiniBatchKMeans":
            k_means_clustering = MiniBatchKMeans(n_clusters=self.n_words,
                                                 init_size=2*self.n_words)
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

    def save(self):
        check_n_make_dir(self.model_path)
        joblib.dump(self.k_means_clustering, self.save_file_cluster)

        with open(self.save_file_parameters, "w") as f:
            j_file = json.dumps(self.parameters)
            f.write(j_file)

    def load(self):
        if os.path.isfile(self.save_file_cluster):
            self.k_means_clustering = joblib.load(self.save_file_cluster)
            self.n_words = self.k_means_clustering.n_clusters
        else:
            self.k_means_clustering = None

        if os.path.isfile(self.save_file_parameters):
            with open(self.save_file_parameters) as json_file:
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
        descriptors = self._remove_empty_desc(descriptors)
        descriptors = np.concatenate(descriptors, axis=0)
        print("Fitting Bag of Words to feature space...")
        print("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        print("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.fit(descriptors)
        print("done in %0.3fs" % (time() - t0))

    def partial_fit(self, descriptors):
        descriptors = np.concatenate(descriptors, axis=0)
        print("Fitting Bag of Words to feature space...")
        print("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        print("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.partial_fit(descriptors)
        print("done in %0.3fs" % (time() - t0))

    def _translate_to_visual_words(self, vector):
        word = self.k_means_clustering.predict(vector)
        return word

    def _normalize_word_bag(self, word_bag):
        if self.parameters["normalize"]:
            sum_words = np.sum(word_bag, axis=1)
            return np.divide(word_bag, sum_words)
        else:
            return word_bag

    def _bag_up_descriptors(self, descriptors):
        word_bag = np.zeros((1, self.n_words))
        if descriptors is not None:
            words = self.k_means_clustering.predict(descriptors)
            for word in words:
                word_bag[0, word] += 1
        return word_bag

    def transform(self, desc_sets):
        word_bags = []
        for descriptors in desc_sets:
            word_bag = self._bag_up_descriptors(descriptors)
            word_bags.append(word_bag)
        return word_bags

