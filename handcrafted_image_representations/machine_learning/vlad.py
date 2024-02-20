import numpy as np
import os
import logging

from time import time
from sklearn.cluster import MiniBatchKMeans, KMeans
import json
import joblib

from handcrafted_image_representations.machine_learning.bag_of_words import remove_empty_desc
from handcrafted_image_representations.utils.utils import check_n_make_dir


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
                                                 init_size=2 * self.n_words,
                                                 n_init="auto")
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

    def fit(self, descriptors):
        self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = remove_empty_desc(descriptors)
        descriptors = np.concatenate(descriptors, axis=0)
        logging.info("Fitting Visual Dictionary (n_words={}) to feature space...".format(self.n_words))
        logging.info("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        logging.info("Each Vector with {} features".format(descriptors.shape[1]))
        self.parameters["n_features"] = descriptors.shape[1]
        t0 = time()
        self.k_means_clustering.fit(descriptors.astype("double"))
        logging.info("done in %0.3fs" % (time() - t0))

    def partial_fit(self, descriptors):
        if self.k_means_clustering is None:
            self.k_means_clustering = self._init_cluster_method(self.cluster_method)
        descriptors = np.concatenate(descriptors, axis=0)
        logging.info("Fitting VLAD to feature space...")
        logging.info("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        logging.info("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.k_means_clustering.partial_fit(descriptors)
        print("done in %0.3fs" % (time() - t0))

    def _translate_to_visual_words(self, vector):
        word = self.k_means_clustering.predict(vector)
        return word

    def transform(self, desc_sets):
        if type(desc_sets) is not list:
            desc_sets = [desc_sets]
        word_bags = []
        for descriptors in desc_sets:
            word_bag = self.transform_single(descriptors)
            word_bags.append(word_bag)
        return word_bags

    def transform_single(self, descriptors):
        if descriptors is None:
            return np.zeros((1, self.n_words*self.parameters["n_features"]))
        descriptors = np.array(descriptors, dtype=np.float64)
        visual_words = self.k_means_clustering.cluster_centers_

        # Step 2: Vector Quantization (Compute residuals)
        labels = self.k_means_clustering.predict(descriptors)
        residuals = [descriptors[i] - visual_words[labels[i]] for i in range(len(descriptors))]

        # Step 3: Aggregation (Sum residuals to get VLAD representation)
        vlad_representation = np.zeros((self.n_words, descriptors.shape[1]))
        for i in range(len(residuals)):
            vlad_representation[labels[i]] += residuals[i]

        # L2-normalization
        vlad_representation = vlad_representation.flatten()
        vlad_representation /= np.sqrt(np.sum(vlad_representation ** 2))
        vlad_representation = np.reshape(vlad_representation, (1, -1))
        return vlad_representation
