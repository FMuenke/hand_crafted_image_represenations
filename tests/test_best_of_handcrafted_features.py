import unittest
from unittest.mock import patch
from handcrafted_image_representations.machine_learning.best_of_handcrafted_features import BestOfHandcraftedFeatures


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestBestOfHandcraftedFeatures(unittest.TestCase):

    def setUp(self):

        self.class_mapping = {
            "0": 0,
            "1": 1,
            # Add more classes
        }

        self.best_of_hcf = BestOfHandcraftedFeatures(
            self.class_mapping,
            feature=["gray-sift", "rgb-sift"],
            image_size={"height": 64, "width": 64},
            sampling_method="one",
            aggregator="global_avg",
            clf_type="lr"
            )

    def test_fit(self):
        model_folder = "./tests/dummy_model_1"
        data_path = "./tests/test_data"
        tag_type = "cls"
        load_all = False
        report_path = model_folder

        with patch('handcrafted_image_representations.machine_learning.feature_extractor.tqdm', notqdm):
            with patch('builtins.print'):
                best_f1_score = self.best_of_hcf.fit_folder(model_folder, data_path, tag_type, load_all, report_path)
                self.assertIsNotNone(best_f1_score)


if __name__ == "__main__":
    unittest.main()
