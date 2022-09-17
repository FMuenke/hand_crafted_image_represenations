import numpy as np
from datastructure.image_handler import ImageHandler
import cv2


class Preprocessor:
    def __init__(self, image_size, normalize=True, mode="norm", resize_option="standard"):
        self.image_size = image_size
        self.min_height = 16
        self.min_width = 16
        self.do_normalize = normalize
        self.mode = mode
        self.resize_option = resize_option

    def standard_resize(self, image, width, height):
        if height is not None and width is not None:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        if width is not None and height is None:
            h, w = image.shape[:2]
            s = width / w
            h_new = int(h * s)
            image = cv2.resize(image, (width, h_new), interpolation=cv2.INTER_CUBIC)

        if height is not None and width is None:
            h, w = image.shape[:2]
            s = height / h
            w_new = int(w * s)
            image = cv2.resize(image, (w_new, height), interpolation=cv2.INTER_CUBIC)

        return image

    def aspect_ratio_preserving_resize(self, image, input_height, input_width):
        height, width = image.shape[:2]
        if height < width:
            image = np.transpose(image, (1, 0, 2))
            image = cv2.flip(image, 1)

        return self.standard_resize(image, height=input_height, width=input_width)

    def normalize(self, image):
        img_h = ImageHandler(image)
        norm = img_h.normalize()
        if self.mode == "norm":
            return norm
        if self.mode == "norm_scale":
            return norm / 255
        raise ValueError("Mode {} not recognised, use norm or norm_scale".format(self.mode))

    def apply(self, image):
        if self.resize_option == "standard":
            image = self.standard_resize(image, width=self.image_size[1], height=self.image_size[0])
        if self.resize_option == "rotate_max_dimension_to_height":
            image = self.aspect_ratio_preserving_resize(
                image, input_height=self.image_size[0], input_width=self.image_size[1]
            )
        if self.do_normalize:
            image = self.normalize(image)
        return image
