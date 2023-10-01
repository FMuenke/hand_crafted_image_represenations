import unittest
import numpy as np
from hand_crafted_image_representations.machine_learning.aggregator import Aggregator


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.descriptors = [np.random.rand(10, 128), np.random.rand(100, 128)]

    def test_fit_transform_bag_of_words(self):
        opt = {"aggregator": "bag_of_words", "complexity": 100}
        aggregator = Aggregator(opt)
        aggregated = aggregator.fit_transform(self.descriptors)
        self.assertIsNotNone(aggregated)

    def test_fit_transform_fisher_vector(self):
        opt = {"aggregator": "fisher_vector", "complexity": 100}
        aggregator = Aggregator(opt)
        aggregated = aggregator.fit_transform(self.descriptors)
        self.assertIsNotNone(aggregated)

    def test_fit_transform_global_avg(self):
        opt = {"aggregator": "global_avg"}
        aggregator = Aggregator(opt)
        aggregated = aggregator.fit_transform(self.descriptors)
        self.assertIsNotNone(aggregated)

    def test_fit_transform_global_max(self):
        opt = {"aggregator": "global_max"}
        aggregator = Aggregator(opt)
        aggregated = aggregator.fit_transform(self.descriptors)
        self.assertIsNotNone(aggregated)

    def test_fit_transform_vlad(self):
        opt = {"aggregator": "vlad", "complexity": 100}
        aggregator = Aggregator(opt)
        aggregated = aggregator.fit_transform(self.descriptors)
        self.assertIsNotNone(aggregated)


if __name__ == '__main__':
    unittest.main()
