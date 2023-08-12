import unittest
from unittest.mock import patch
from classic_image_classification.machine_learning.best_of_bag_of_words import BestOfBagOfWords


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestBestOfBagOfWords(unittest.TestCase):

    def setUp(self):
        self.opt = {
            "data_split_mode": "fixed",
            "feature": ["gray-sift", "rgb-sift"],
            "image_size": {"height": 64, "width": 64},
            "sampling_method": "one",
            "aggregator": "global_avg",
            "type": "lr"
        }

        self.class_mapping = {
            "0": 0,
            "1": 1,
            # Add more classes
        }

        self.best_of_bag_of_words = BestOfBagOfWords(self.opt, self.class_mapping)

    def test_fit(self):
        model_folder = "./tests/dummy_model"
        data_path = "./tests/test_data"
        tag_type = "cls"
        load_all = False
        report_path = "path/to/reports"

        with patch('classic_image_classification.machine_learning.feature_extractor.tqdm', notqdm):
            with patch('builtins.print'):
                best_f1_score = self.best_of_bag_of_words.fit(model_folder, data_path, tag_type, load_all, report_path)
                self.assertIsNotNone(best_f1_score)


if __name__ == "__main__":
    unittest.main()
