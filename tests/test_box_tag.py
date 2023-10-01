import unittest
import numpy as np
from handcrafted_image_representations.data_structure.box_tag import BoxTag


class TestBoxTag(unittest.TestCase):

    def test_load_data(self):
        # Create a dummy image
        image_path = "./tests/test_data/images/img_0.jpg"

        # Create a BoxTag instance
        tag = BoxTag(tag_id=1, path_to_image=image_path, tag_class=["class1"], box=[0, 0, 0, 20, 20],
                     class_mapping=None)

        # Load data and check shape
        data = tag.load_data()
        self.assertEqual(data.shape, (20, 20, 3))

    def test_load_y(self):
        # Create a BoxTag instance
        tag = BoxTag(tag_id=1, path_to_image="./tests/test_data/images/img_0.jpg", tag_class=["class1"],
                     box=[0, 10, 0, 20, 20],
                     class_mapping={"class1": 0})

        # Load y
        y = tag.load_y()
        self.assertEqual(y, np.array(0))

    def test_has_relevant_classes(self):
        # Create a BoxTag instance
        tag = BoxTag(tag_id=1, path_to_image="./tests/test_data/images/img_0.jpg",
                     tag_class=["class1"], box=["0", 10, 0, 20, 20],
                     class_mapping={"class1": 0})

        # Check relevant class
        self.assertTrue(tag.has_relevant_classes(["class1"]))

        # Check irrelevant class
        self.assertFalse(tag.has_relevant_classes(["class2"]))


if __name__ == '__main__':
    unittest.main()
