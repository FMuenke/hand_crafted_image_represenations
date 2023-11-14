import unittest
from unittest.mock import patch

from handcrafted_image_representations.machine_learning.image_classifier import ImageClassifier


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestImageClassifier(unittest.TestCase):

    def setUp(self):

        self.path = "./tests/test_data/"

        self.opt = {
            "feature": "gray-sift",
            "image_size_width": 64,
            "image_size_height": 64,
            "sampling_method": "dense",
            "sampling_step": 8,
            "sampling_window": 8,
            "aggregator": "bag_of_words",
            "complexity": 8,
            "clf_type": "lr"
        }

        self.class_mapping = {"0": 0, "1": 1}
        self.classifier = ImageClassifier(self.class_mapping, **self.opt)

    def test_new(self):
        self.classifier.new()
        self.assertIsNotNone(self.classifier.feature_extractor)
        self.assertIsNotNone(self.classifier.aggregator)
        self.assertIsNotNone(self.classifier.classifier)

    def test_fit_and_eval(self):
        with patch('handcrafted_image_representations.machine_learning.feature_extractor.tqdm', notqdm):
            with patch('handcrafted_image_representations.machine_learning.image_classifier.tqdm', notqdm):
                with patch('builtins.print'):
                    self.classifier.fit_folder(data_path=self.path, tag_type="cls")
                    self.classifier.evaluate(data_path=self.path, tag_type="cls")


if __name__ == "__main__":
    unittest.main()
