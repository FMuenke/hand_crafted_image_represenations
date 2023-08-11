import unittest
import numpy as np
from classic_image_classification.machine_learning.bag_of_words import BagOfWords


class TestBagOfWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.descriptor_dim = 128
        cls.n_words = 4
        cls.descriptor_shape = (10, cls.descriptor_dim)
        cls.descriptors = np.random.rand(*cls.descriptor_shape)

    def test_fit(self):
        bow = BagOfWords(n_words=self.n_words)
        bow.fit([self.descriptors, None])
        self.assertTrue(bow.is_fitted())

    def test_transform(self):
        bow = BagOfWords(n_words=self.n_words)
        bow.fit([self.descriptors])
        transformed = bow.transform([self.descriptors, None])
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0].shape, (1, self.n_words))

    def test_normalize_word_bag(self):
        bow = BagOfWords(n_words=self.n_words, normalize=True)
        word_bag = np.random.rand(1, self.n_words)
        normalized_word_bag = bow._normalize_word_bag(word_bag)
        self.assertTrue(np.allclose(np.sum(normalized_word_bag, axis=1), 1))


if __name__ == "__main__":
    unittest.main()
