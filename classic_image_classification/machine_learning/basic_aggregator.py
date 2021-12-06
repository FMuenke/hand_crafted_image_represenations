import numpy as np


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
        print("Basic Aggregator is fitting...")
        descriptor = descriptors[0]
        self.n_features = descriptor.shape[1]
        print("Basic Aggregator was fitted. Descriptors with {} features".format(descriptor.shape[1]))
