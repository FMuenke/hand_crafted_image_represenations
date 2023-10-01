import unittest
import os
from handcrafted_image_representations.data_structure.data_set import DataSet


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data_set_dir = "./tests/test_data/"
        self.tag_type = "cls"
        self.class_mapping = {
            "0": 0,
            "1": 1
        }
        self.data_set = DataSet(self.data_set_dir, self.tag_type, self.class_mapping)

    def test_load_labeled_image(self):
        args = (self.data_set_dir, "img_0.jpg")
        tags = self.data_set.load_labeled_image(args)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0].image_id, os.path.join(self.data_set_dir, "images", "img_0.jpg"))

    def test_load_directory(self):
        self.data_set.load_directory(self.data_set_dir)
        self.assertTrue(len(self.data_set.tags) > 0)

    def test_load_data(self):
        self.assertTrue(len(self.data_set.tags) > 0)

    def test_get_tags_all_classes(self):
        tags = self.data_set.get_tags()
        self.assertTrue(len(tags) > 0)

    def test_get_tags_specific_classes(self):
        tags = self.data_set.get_tags(classes_to_consider=["0"])
        self.assertTrue(len(tags) > 0)


if __name__ == '__main__':
    unittest.main()
