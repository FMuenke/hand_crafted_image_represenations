import joblib
import logging
from time import time
import os
import numpy as np
import json
from sklearn.mixture import GaussianMixture

from classic_image_classification.utils.utils import check_n_make_dir


class FisherVector:
    def __init__(self, n_components=100):
        self.gmm = None
        self.n_components = n_components
        self.parameters = dict()

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

    def save(self, model_path):
        save_file_gmm = os.path.join(model_path, "gaussian_mixture.pkl")
        check_n_make_dir(model_path)
        joblib.dump(self.gmm, save_file_gmm)

        save_file_parameters = os.path.join(model_path, "parameters.json")

        with open(save_file_parameters, "w") as f:
            j_file = json.dumps(self.parameters)
            f.write(j_file)

    def load(self, model_path):
        save_file_gmm = os.path.join(model_path, "gaussian_mixture.pkl")
        if os.path.isfile(save_file_gmm):
            self.gmm = joblib.load(save_file_gmm)
            self.n_components = self.gmm.n_components
        else:
            self.gmm = None

        save_file_parameters = os.path.join(model_path, "parameters.json")
        if os.path.isfile(save_file_parameters):
            with open(save_file_parameters) as json_file:
                self.parameters = json.load(json_file)

    def fit(self, descriptors):
        descriptors = self._remove_empty_desc(descriptors)
        descriptors = np.concatenate(descriptors, axis=0)
        logging.info("Fitting Gaussian Mixture Model to feature space...")
        logging.info("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        logging.info("Each Vector with {} features".format(descriptors.shape[1]))
        self.parameters["n_features"] = descriptors.shape[1]
        t0 = time()
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=10000
        )
        self.gmm.fit(descriptors)
        logging.info("done in %0.3fs" % (time() - t0))

    def fisher_vector(self, descriptors):
        if descriptors is None:
            return np.zeros((1, self.n_components * self.parameters["n_features"] * 2))
        descriptor_dim = descriptors.shape[1]

        # Compute the responsibilities of each descriptor to each GMM component
        responsibilities = self.gmm.predict_proba(descriptors)

        # Compute the means and covariances of the GMM components
        means = self.gmm.means_
        covariances = self.gmm.covariances_

        # Compute the FV using the Fisher Vector formula
        fv = np.zeros((self.n_components, 2 * descriptor_dim))
        for i in range(self.n_components):
            d_mu = descriptors - means[i]
            prec_factor = np.sqrt(self.gmm.precisions_[i])

            # Computing the first order term (mean derivatives)
            fv[i, :descriptor_dim] = (responsibilities[:, i].dot(d_mu) * prec_factor).sum(axis=0)

            # Computing the second order term (covariance derivatives)
            fv[i, descriptor_dim:] = np.sum(responsibilities[:, i, np.newaxis] * d_mu * prec_factor, axis=0)

            # Normalization
            fv[i] /= (np.sqrt(np.sum(fv[i] ** 2)) + 1e-5)

        # Concatenate and normalize the Fisher Vector
        fv = fv.flatten()
        return np.reshape(fv, (1, -1))

    def transform(self, desc_sets):
        assert self.gmm is not None, "Abort: Fit or load Model First"

        fisher_vectors = []
        for descriptors in desc_sets:
            fv = self.fisher_vector(descriptors)
            fisher_vectors.append(fv)
        return fisher_vectors

    def is_fitted(self):
        if self.gmm is not None:
            return True
        else:
            return False
