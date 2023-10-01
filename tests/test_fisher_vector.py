import unittest
import numpy as np
from handcrafted_image_representations.machine_learning.fisher_vector import FisherVector


class TestFisherVector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptor_dim = 128
        cls.n_components = 4
        cls.descriptor_shape = (10, cls.descriptor_dim)
        cls.descriptors = np.random.rand(*cls.descriptor_shape)

    def test_fit(self):
        fv = FisherVector(n_components=self.n_components)
        fv.fit([self.descriptors, None])
        self.assertTrue(fv.is_fitted())

    def test_transform(self):
        fv = FisherVector(n_components=self.n_components)
        fv.fit([self.descriptors])
        transformed = fv.transform([self.descriptors, None])
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0].shape, (1, self.n_components * self.descriptor_dim * 2))

    def test_fisher_vector(self):
        fv = FisherVector(n_components=self.n_components)
        fv.fit([self.descriptors])
        transformed = fv.fisher_vector(self.descriptors)
        self.assertEqual(transformed.shape, (1, self.n_components * self.descriptor_dim * 2))


if __name__ == "__main__":
    unittest.main()
