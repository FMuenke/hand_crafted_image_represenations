from machine_learning.fisher_vector import FisherVector
from machine_learning.bag_of_words import BagOfWords
from machine_learning.basic_aggregator import BasicAggregator


class AggregatorHandler:
    def __init__(self, model_path, opt):
        self.model_path = model_path
        self.opt = opt["classifier_opt"]

        self.aggregator = None

    def is_fitted(self):
        if self.aggregator is None:
            return False
        else:
            return self.aggregator.is_fitted()

    def fit_transform(self, descriptors):
        self.fit(descriptors)
        return self.transform(descriptors)

    def transform(self, descriptors):
        return self.aggregator.transform(descriptors)

    def fit(self, descriptors):
        self.new_aggregator()
        self.aggregator.fit(descriptors)

    def new_aggregator(self):
        if "bag_of_words" == self.opt["aggregator"]:
            self.aggregator = BagOfWords(model_path=self.model_path,
                                         n_words=self.opt["complexity"])

        if "fisher_vector" == self.opt["aggregator"]:
            self.aggregator = FisherVector(model_path=self.model_path,
                                           n_components=self.opt["complexity"])

        if "basic_mean" == self.opt["aggregator"]:
            self.aggregator = BasicAggregator(model_path=self.model_path)
