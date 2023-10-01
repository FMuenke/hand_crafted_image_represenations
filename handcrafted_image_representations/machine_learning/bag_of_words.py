import joblib
import json
import os
from time import time
import numpy as np
import logging

from sklearn.cluster import KMeans, MiniBatchKMeans
from handcrafted_image_representations.utils.utils import check_n_make_dir


def remove_empty_desc(descriptors):
    """
    Empty descriptors can not be used to find visual words
    and are deleted before building visual word dictionary
    Args:
        descriptors: list of descriptors sets per image can contain None

    Returns:
        filtered list of descriptors sets per image CANNOT contain None
    """
    descriptors_out = []
    for desc in descriptors:
        if desc is None:
            continue
        descriptors_out.append(desc)
    return descriptors_out


class BagOfWords:
    def __init__(self, n_words=100, cluster_method="MiniBatchKMeans", normalize=False):

        self.n_words = n_words
        self.parameters = dict()
        self.cluster_method = cluster_method
        self.parameters["normalize"] = normalize

        self.k_means_clustering = None

    def _init_cluster_method(self, cluster_method):
        if cluster_method == "MiniBatchKMeans":
            k_means_clustering = MiniBatchKMeans(
                n_init="auto",
                n_clusters=self.n_words,
                init_size=2*self.n_words
            )
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
        save_file_cluster = os.path.join(model_path, "bag_of_words.pkl")
        save_file_parameters = os.path.join(model_path, "parameters.json")
        joblib.dump(self.k_means_clustering, save_file_cluster)

        with open(save_file_parameters, "w") as f:
            j_file = json.dumps(self.parameters)
            f.write(j_file)

    def load(self, model_path):
        save_file_cluster = os.path.join(model_path, "bag_of_words.pkl")
        save_file_parameters = os.path.join(model_path, "parameters.json")
        if os.path.isfile(save_file_cluster):
            self.k_means_clustering = joblib.load(save_file_cluster)
            self.n_words = self.k_means_clustering.n_clusters
        else:
            self.k_means_clustering = None

        if os.path.isfile(save_file_parameters):
            with open(save_file_parameters) as json_file:
                self.parameters = json.load(json_file)

    def fit(self, descriptors):
        self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = remove_empty_desc(descriptors)
        descriptors = np.concatenate(descriptors, axis=0)
        logging.info("Fitting Bag of Words (n_words={}) to feature space...".format(self.n_words))
        logging.info("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        logging.info("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.fit(descriptors.astype("double"))
        logging.info("done in %0.3fs" % (time() - t0))

    def partial_fit(self, descriptors):
        if self.k_means_clustering is None:
            self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = np.concatenate(descriptors, axis=0)
        logging.info("Fitting Bag of Words to feature space...")
        logging.info("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        logging.info("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.partial_fit(descriptors)
        logging.info("done in %0.3fs" % (time() - t0))

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
            descriptors = descriptors.astype("double")
            words = self.k_means_clustering.predict(descriptors)
            for word in words:
                word_bag[0, word] += 1
        return word_bag

    def transform(self, desc_sets):
        if type(desc_sets) is not list:
            desc_sets = [desc_sets]
        word_bags = []
        for descriptors in desc_sets:
            word_bag = self._bag_up_descriptors(descriptors)
            word_bags.append(word_bag)
        return word_bags
