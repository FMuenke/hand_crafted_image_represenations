import unittest
import numpy as np
from unittest.mock import Mock
from handcrafted_image_representations.machine_learning.compact_vlad import CompactVLAD


class TestCompactVLAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptor_dim = 128
        cls.n_words = 4
        cls.descriptor_shape = (10, cls.descriptor_dim)
        cls.descriptors = np.random.rand(*cls.descriptor_shape)

    def test_fit(self):
        vlad = CompactVLAD(n_words=self.n_words)
        vlad.fit([self.descriptors, None])
        self.assertTrue(vlad.is_fitted())

    def test_transform_single(self):
        vlad = CompactVLAD(n_words=self.n_words)
        vlad.k_means_clustering = Mock()
        vlad.k_means_clustering.cluster_centers_ = np.random.rand(self.n_words, self.descriptor_dim)
        vlad.k_means_clustering.predict.return_value = np.random.randint(0, self.n_words, self.descriptor_shape[0])
        transformed = vlad.transform_single(self.descriptors)
        self.assertEqual(transformed.shape, (1, self.n_words))

    def test_transform(self):
        vlad = CompactVLAD(n_words=self.n_words)
        vlad.fit([self.descriptors])
        transformed = vlad.transform([self.descriptors, None])
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0].shape, (1, self.n_words))

        transformed = vlad.transform(self.descriptors)
        self.assertEqual(len(transformed), 1)

if __name__ == "__main__":
    unittest.main()
