import unittest
import numpy as np
from unittest.mock import Mock
from handcrafted_image_representations.machine_learning.vlad import VLAD


class TestVLAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptor_dim = 128
        cls.n_words = 4
        cls.descriptor_shape = (10, cls.descriptor_dim)
        cls.descriptors = np.random.rand(*cls.descriptor_shape)

    def test_fit(self):
        vlad = VLAD(n_words=self.n_words)
        vlad.fit([self.descriptors, None])
        self.assertTrue(vlad.is_fitted())

    def test_transform_single(self):
        vlad = VLAD(n_words=self.n_words)
        vlad.k_means_clustering = Mock()
        vlad.k_means_clustering.cluster_centers_ = np.random.rand(self.n_words, self.descriptor_dim)
        vlad.k_means_clustering.predict.return_value = np.random.randint(0, self.n_words, self.descriptor_shape[0])
        transformed = vlad.transform_single(self.descriptors)
        self.assertEqual(transformed.shape, (1, self.n_words * self.descriptor_dim))

    def test_transform(self):
        vlad = VLAD(n_words=self.n_words)
        vlad.fit([self.descriptors])
        transformed = vlad.transform([self.descriptors, None])
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0].shape, (1, self.n_words * self.descriptor_dim))


if __name__ == "__main__":
    unittest.main()
