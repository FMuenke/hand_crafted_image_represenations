import unittest
import os
from hand_crafted_image_representations.data_structure.labeled_image import get_file_name, LabeledImage


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self.base_path = "./tests/test_data/images"

    def test_get_file_name(self):
        extensions = [".jpg", ".png"]
        expected_file = "./tests/test_data/images/img_0.jpg"

        result = get_file_name(self.base_path, "img_0", extensions)
        self.assertEqual(result, expected_file)


class TestLabeledImage(unittest.TestCase):

    def setUp(self):
        self.base_path = "./tests/test_data"
        self.data_id = "img_0"

    def test_init_with_image(self):
        labeled_img = LabeledImage(self.base_path, self.data_id)
        self.assertEqual(labeled_img.image_file, os.path.join(self.base_path, "images", "img_0.jpg"))

    def test_init_with_label(self):
        labeled_img = LabeledImage(self.base_path, self.data_id)
        self.assertEqual(labeled_img.label_file, os.path.join(self.base_path, "labels", "img_0.txt"))

    def test_get_image_size(self):
        labeled_img = LabeledImage(self.base_path, self.data_id)
        result = labeled_img.get_image_size()
        self.assertEqual(result, (128, 128))


if __name__ == "__main__":
    unittest.main()
