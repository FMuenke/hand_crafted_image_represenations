import unittest
import numpy as np
from unittest.mock import patch
from hand_crafted_image_representations.machine_learning.classic_image_classifier import ClassicImageClassifier


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestClassicImageClassifier(unittest.TestCase):

    def setUp(self):

        self.path = "./tests/test_data/"

        self.opt = {
            "feature": "gray-sift",
            "image_size": {"height": 64, "width": 64},
            "sampling_method": "dense",
            "sampling_step": 8,
            "sampling_window": 8,
            "aggregator": "bag_of_words",
            "complexity": 8,
            "type": "lr"
        }

        self.class_mapping = {"0": 0, "1": 1}
        self.classifier = ClassicImageClassifier(self.opt, self.class_mapping)

    def test_new(self):
        self.classifier.new()
        self.assertIsNotNone(self.classifier.feature_extractor)
        self.assertIsNotNone(self.classifier.aggregator)
        self.assertIsNotNone(self.classifier.classifier)

    def test_fit_and_eval(self):
        with patch('hand_crafted_image_representations.machine_learning.feature_extractor.tqdm', notqdm):
            with patch('hand_crafted_image_representations.machine_learning.classic_image_classifier.tqdm', notqdm):
                with patch('builtins.print'):
                    self.classifier.fit(data_path=self.path, tag_type="cls")
                    self.classifier.evaluate(data_path=self.path, tag_type="cls")


if __name__ == "__main__":
    unittest.main()
