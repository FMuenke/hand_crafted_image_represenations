import unittest
import numpy as np
from unittest.mock import patch
from classic_image_classification.machine_learning.feature_extractor import FeatureExtractor
from classic_image_classification.data_structure.box_tag import BoxTag


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.features_to_use = "gray-sift"
        self.normalize = True
        self.sampling_method = "dense"
        self.sampling_steps = 20
        self.sampling_window = 20
        self.image_height = 256
        self.image_width = 256
        self.resize_option = "standard"

        self.feature_extractor = FeatureExtractor(
            features_to_use=self.features_to_use,
            normalize=self.normalize,
            sampling_method=self.sampling_method,
            sampling_steps=self.sampling_steps,
            sampling_window=self.sampling_window,
            image_height=self.image_height,
            image_width=self.image_width,
            resize_option=self.resize_option
        )

    def test_extract_x(self):
        mock_image = np.random.random((128, 128, 3))  # Mock an image
        with patch('cv2.resize', return_value=mock_image):
            x = self.feature_extractor.extract_x(mock_image)
            self.assertEqual(x.shape, (25, 128))

    def test_extract_trainings_data(self):
        mock_tag_data = np.random.random((128, 128, 3))  # Mock tag data
        mock_tags = {1: BoxTag("00", "./tests/test_data/images/img_0.jpg", "0", ["0", 0, 0, 128, 128], {"0": 0, "1": 1})}
        with patch('cv2.resize', return_value=mock_tag_data):
            with patch('classic_image_classification.machine_learning.feature_extractor.tqdm', notqdm):
                with patch('builtins.print'):
                    x, y = self.feature_extractor.extract_trainings_data(mock_tags)
                    self.assertEqual(len(x), 1)
                    self.assertEqual(len(y), 1)


if __name__ == '__main__':
    unittest.main()
