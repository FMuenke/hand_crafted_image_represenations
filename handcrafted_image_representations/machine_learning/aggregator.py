from handcrafted_image_representations.machine_learning.fisher_vector import FisherVector
from handcrafted_image_representations.machine_learning.bag_of_words import BagOfWords
from handcrafted_image_representations.machine_learning.global_aggregator import GlobalAggregator
from handcrafted_image_representations.machine_learning.vlad import VLAD
from handcrafted_image_representations.machine_learning.compact_vlad import CompactVLAD


def format_aggregator_settings(opt):
    if opt["aggregator"] in ["global_avg", "global_max"]:
        opt["complexity"] = None
    return opt


class Aggregator:
    def __init__(self, opt):
        self.opt = format_aggregator_settings(opt)

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
        elif "bag_of_words_tfidf" == self.opt["aggregator"]:
            self.aggregator = BagOfWords(n_words=self.opt["complexity"], tf_idf=True)
        elif "fisher_vector" == self.opt["aggregator"]:
            self.aggregator = FisherVector(n_components=self.opt["complexity"])
        elif self.opt["aggregator"] in ["global_avg", "glob_avg", "global_max", "glob_max"]:
            self.aggregator = GlobalAggregator(self.opt["aggregator"])
        elif "vlad" == self.opt["aggregator"]:
            self.aggregator = VLAD(n_words=self.opt["complexity"])
        elif "c_vlad" == self.opt["aggregator"]:
            self.aggregator = CompactVLAD(n_words=self.opt["complexity"])
        else:
            raise Exception("Unknown Option for Aggregator: {}".format(self.opt["aggregator"]))

    def save(self, path):
        self.aggregator.save(path)

    def load(self, path):
        self.new()
        self.aggregator.load(path)
