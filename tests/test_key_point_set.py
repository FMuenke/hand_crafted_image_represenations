import unittest
import cv2
import numpy as np
from hand_crafted_image_representations.machine_learning.key_point_set import (
    key_points_to_open_cv_key_points,
    open_cv_key_points_to_key_points,
    build_open_cv_key_point_detectors,
    get_one_key_point,
    KeyPointSet
)


class TestKeyPoints(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 255, size=(300, 300)).astype(np.uint8)

    def test_key_points_to_open_cv_key_points(self):
        key_points = [[100, 100, 10], [150, 200, 15]]
        open_cv_key_points = key_points_to_open_cv_key_points(key_points)

        self.assertIsInstance(open_cv_key_points[0], cv2.KeyPoint)
        self.assertEqual(len(open_cv_key_points), len(key_points))

    def test_open_cv_key_points_to_key_points(self):
        open_cv_key_points = [cv2.KeyPoint(100, 100, 10), cv2.KeyPoint(150, 200, 15)]
        key_points = open_cv_key_points_to_key_points(open_cv_key_points)

        self.assertIsInstance(key_points[0], list)
        self.assertEqual(len(key_points), len(open_cv_key_points))

    def test_build_open_cv_key_point_detectors(self):
        detectors = build_open_cv_key_point_detectors()

        self.assertIn("orb", detectors)
        self.assertIn("sift", detectors)
        self.assertIn("akaze", detectors)
        self.assertIn("kaze", detectors)
        self.assertIn("brisk", detectors)

    def test_get_one_key_point(self):
        key_point = get_one_key_point(self.image)

        self.assertEqual(len(key_point), 1)
        self.assertEqual(len(key_point[0]), 3)

    def test_key_point_set_dense(self):
        kp_set = KeyPointSet("dense", sampling_steps=50, sampling_window=10)
        key_points = kp_set.get_key_points(self.image)

        self.assertTrue(isinstance(key_points, list))
        self.assertTrue(isinstance(key_points[0], list))

    def test_key_point_set_one(self):
        kp_set = KeyPointSet("one")
        key_points = kp_set.get_key_points(self.image)

        self.assertTrue(isinstance(key_points, list))
        self.assertTrue(isinstance(key_points[0], list))
        self.assertEqual(len(key_points), 1)

    def test_key_point_set_open_cv(self):
        for sampling in ["sift", "kaze", "orb"]:
            kp_set = KeyPointSet(sampling)
            image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)  # Convert to color image
            open_cv_key_points = kp_set.get_open_cv_key_points(image)
            self.assertTrue(isinstance(open_cv_key_points, tuple))
            self.assertTrue(isinstance(open_cv_key_points[0], cv2.KeyPoint))


if __name__ == '__main__':
    unittest.main()
