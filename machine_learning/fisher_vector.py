import joblib
from time import time
import os
import numpy as np
from sklearn.mixture import GaussianMixture

from utils.utils import check_n_make_dir


class FisherVector:
    def __init__(self, model_path, n_components=100):
        self.model_path = model_path
        self.save_file_gmm = os.path.join(self.model_path, "gaussian_mixture.pkl")
        self.gmm = None
        self.n_components = n_components

    def save(self):
        check_n_make_dir(self.model_path)
        joblib.dump(self.gmm, self.save_file_gmm)

    def load(self):
        if os.path.isfile(self.save_file_gmm):
            self.gmm = joblib.load(self.save_file_gmm)
            self.n_components = self.gmm.n_components
        else:
            self.gmm = None

    def fit(self, descriptors):
        descriptors = np.concatenate(descriptors, axis=0)
        print("Fitting Gaussian Mixture Model to feature space...")
        print("Feature Vectors to be fitted: {}".format(descriptors.shape[0]))
        print("Each Vector with {} features".format(descriptors.shape[1]))
        t0 = time()
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type='diag')
        self.gmm.fit(descriptors)
        print("done in %0.3fs" % (time() - t0))

    def fisher_vector(self, descriptors):
        x = np.atleast_2d(descriptors)
        N = x.shape[0]
        # Compute posterior probabilities.
        Q = self.gmm.predict_proba(x)  # NxK
        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_x = np.dot(Q.T, x) / N
        Q_x_2 = np.dot(Q.T, x ** 2) / N
        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - self.gmm.weights_
        d_mu = Q_x - Q_sum * self.gmm.means_
        d_sigma = (
                - Q_x_2
                - Q_sum * self.gmm.means_ ** 2
                + Q_sum * self.gmm.covariances_
                + 2 * Q_x * self.gmm.means_)

        # Merge derivatives into a vector.
        fv = np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
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
