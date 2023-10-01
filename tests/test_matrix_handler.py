import unittest
import numpy as np
from hand_crafted_image_representations.data_structure.matrix_handler import MatrixHandler


class TestMatrixHandler(unittest.TestCase):

    def setUp(self):
        self.matrix = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])

    def test_normalize(self):
        matrix_handler = MatrixHandler(self.matrix)
        normalized_matrix = matrix_handler.normalize()
        self.assertLessEqual(np.max(normalized_matrix), 1)
        self.assertGreaterEqual(np.min(normalized_matrix), 0)

    def test_global_pooling(self):
        matrix_handler = MatrixHandler(self.matrix)
        pooled_matrix = matrix_handler.global_pooling("max")
        self.assertEqual(pooled_matrix.shape, (1, 1))

    def test_pooling(self):
        matrix_handler = MatrixHandler(self.matrix)
        down_sampled_matrix = matrix_handler.pooling("max", 2)
        self.assertEqual(down_sampled_matrix.shape, (2, 2))

    def test_apply_convolution(self):
        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])
        matrix_handler = MatrixHandler(self.matrix)
        convolved_matrix = matrix_handler.apply_convolution(kernel)
        self.assertEqual(convolved_matrix.shape, self.matrix.shape)

    def test_cut_roi(self):
        matrix_handler = MatrixHandler(self.matrix)
        roi = matrix_handler.cut_roi((1, 1), 2)
        self.assertEqual(roi.shape, (2, 2))

    def test_merge(self):
        matrix_handler = MatrixHandler(np.stack([self.matrix, self.matrix, self.matrix], axis=2))
        merged_matrix = matrix_handler.merge("max")
        self.assertEqual(merged_matrix.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
