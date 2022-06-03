from classic_image_classification.machine_learning.fisher_vector import FisherVector
from classic_image_classification.machine_learning.bag_of_words import BagOfWords
from classic_image_classification.machine_learning.basic_aggregator import BasicAggregator
from classic_image_classification.machine_learning.vlad import VLAD


class Aggregator:
    def __init__(self, opt):
        self.opt = opt

        self.aggregator = None

    def __str__(self):
        if "complexity" in self.opt:
            return "Aggregator: {} - {}".format(self.opt["aggregator"], self.opt["complexity"])
        else:
            return "Aggregator: {}".format(self.opt["aggregator"])

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
        self.new()
        self.aggregator.fit(descriptors)

    def new(self):
        if "bag_of_words" == self.opt["aggregator"]:
            self.aggregator = BagOfWords(n_words=self.opt["complexity"])

        if "fisher_vector" == self.opt["aggregator"]:
            self.aggregator = FisherVector(n_components=self.opt["complexity"])

        if "basic_mean" == self.opt["aggregator"]:
            self.aggregator = BasicAggregator()

        if "vlad" == self.opt["aggregator"]:
            self.aggregator = VLAD(n_words=self.opt["complexity"])

    def save(self, path):
        self.aggregator.save(path)

    def load(self, path):
        self.new()
        self.aggregator.load(path)
