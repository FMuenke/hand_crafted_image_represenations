import unittest
import numpy as np
from hand_crafted_image_representations.data_structure.image_handler import ImageHandler


class TestImageHandler(unittest.TestCase):

    def setUp(self):
        # Create a sample image
        self.image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_BGR(self):
        img_handler = ImageHandler(self.image)
        bgr_image = img_handler.BGR()
        self.assertTrue(np.array_equal(bgr_image, self.image))

    def test_RGB(self):
        img_handler = ImageHandler(self.image)
        rgb_image = img_handler.RGB()
        self.assertTrue(np.array_equal(rgb_image, self.image[:, :, [2, 1, 0]]))

    def test_gray(self):
        img_handler = ImageHandler(self.image)
        gray_image = img_handler.gray()
        self.assertEqual(gray_image.shape, (100, 100))

    def test_integral(self):
        img_handler = ImageHandler(self.image)
        integral_img = img_handler.integral()
        self.assertEqual(integral_img.shape, (100, 100))


if __name__ == '__main__':
    unittest.main()
