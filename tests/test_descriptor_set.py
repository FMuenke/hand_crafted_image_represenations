import unittest
import numpy as np
from handcrafted_image_representations.machine_learning.descriptor_set import DescriptorSet
from handcrafted_image_representations.machine_learning.key_point_set import KeyPointSet


class TestDescriptorSet(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 255, size=(300, 300, 3)).astype(np.uint8)
        self.key_point_set = KeyPointSet("dense", sampling_steps=50, sampling_window=10)

    def test_compute_lbp(self):
        descriptor_set = DescriptorSet("opponent-lbp+24+7")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_compute_lm(self):
        descriptor_set = DescriptorSet("gray-lm")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_compute_histogram(self):
        descriptor_set = DescriptorSet("rgb-histogram+64")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_compute_hog(self):
        descriptor_set = DescriptorSet("gray-hog+SMALL+L2HYS")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_compute_cv_descriptor_sift(self):
        descriptor_set = DescriptorSet("RGB-sift")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_compute_cv_descriptor_orb(self):
        descriptor_set = DescriptorSet("RGB-orb")
        descriptors = descriptor_set.compute(self.image, self.key_point_set)
        self.assertIsNotNone(descriptors)

    def test_invalid_descriptor_type(self):
        descriptor_set = DescriptorSet("invalid_descriptor")
        with self.assertRaises(ValueError):
            descriptor_set.compute(self.image, self.key_point_set)


if __name__ == '__main__':
    unittest.main()
