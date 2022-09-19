import numpy as np
import logging
from classic_image_classification.utils.utils import check_n_make_dir, save_dict, load_dict
import os


class BasicAggregator:
    def __init__(self):
        self.n_features = None

    def is_fitted(self):
        if self.n_features is None:
            return False
        else:
            return True

    def transform(self, desc_sets):
        aggregates = []

        for descriptors in desc_sets:
            if descriptors is None:
                x = np.zeros((1, self.n_features))
            elif len(descriptors) == 1:
                x = np.zeros((1, self.n_features))
                x[0, :] = descriptors[0]
            else:
                descriptors = np.stack(descriptors, axis=0)
                descriptors = np.mean(descriptors, axis=0)
                x = np.zeros((1, self.n_features))
                x[0, :] = descriptors
            aggregates.append(x)
        return aggregates

    def fit(self, descriptors):
        logging.info("Basic Aggregator is fitting...")
        descriptor = descriptors[0]
        self.n_features = descriptor.shape[1]
        logging.info("Basic Aggregator was fitted. Descriptors with {} features".format(descriptor.shape[1]))

    def save(self, model_path):
        check_n_make_dir(model_path)
        save_file_parameters = os.path.join(model_path, "aggregator-parameters.json")
        save_dict({"n_features": int(self.n_features)}, save_file_parameters)

    def load(self, model_path):
        save_file_parameters = os.path.join(model_path, "aggregator-parameters.json")
        param = load_dict(save_file_parameters)
        self.n_features = int(param["n_features"])
