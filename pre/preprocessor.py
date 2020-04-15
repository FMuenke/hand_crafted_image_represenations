import numpy as np
from datastructure.image_handler import ImageHandler
import cv2


class Preprocessor:
    def __init__(self, image_size, normalize=True, padding=False, mode="norm"):
        self.image_size = image_size
        self.min_height = 16
        self.min_width = 16
        self.do_normalize = normalize
        self.do_padding = padding
        self.mode = mode

        self.obox = None

    def resize(self, image):
        img_h = ImageHandler(image)
        if None in self.image_size:
            height, width = image.shape[:2]
            if height < self.min_height:
                height = self.min_height

            if width < self.min_width:
                width = self.min_width
            return img_h.resize(height=height, width=width)
        return img_h.resize(height=self.image_size[0], width=self.image_size[1])

    def pad(self, image):
        img_h = ImageHandler(image)
        height, width = image.shape[:2]
        if None in self.image_size:
            height, width = image.shape[:2]
            if height < self.min_height:
                height = self.min_height

            if width < self.min_width:
                width = self.min_width
            return img_h.resize(height=height, width=width)
        if height > width:
            if height >= self.image_size[0]:
                new_height = self.image_size[0]
                new_width = int(width * new_height / height)
                image = img_h.resize(height=new_height, width=new_width)
        else:
            if width >= self.image_size[1]:
                new_width = self.image_size[1]
                new_height = int(height * new_width / width)
                image = img_h.resize(height=new_height, width=new_width)
        ih, iw = image.shape[:2]
        ph, pw = self.image_size[0], self.image_size[1]
        x = np.mean(image) * np.ones((ph, pw, 3))
        sy1 = int(ph/2)-int(ih/2)
        sx1 = int(pw/2)-int(iw/2)
        self.obox = [sx1, sy1, sx1+iw, sy1+ih]
        x[sy1:sy1+ih, sx1:sx1+iw, :] = image
        return x

    def normalize(self, image):
        img_h = ImageHandler(image)
        norm = img_h.normalize()
        if self.mode == "norm":
            return norm
        if self.mode == "norm_scale":
            return norm / 255
        raise ValueError("Mode {} not recognised, use norm or norm_scale".format(self.mode))

    def apply(self, image):
        if self.do_padding:
            image = self.pad(image)
        else:
            image = self.resize(image)
        if self.do_normalize:
            image = self.normalize(image)
        return image

    def un_apply(self, image, height, width):
        b, h, w, c = image.shape
        if self.obox is not None:
            image = image[0, :, :, :]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
            image = image[self.obox[1]:self.obox[3], self.obox[0]:self.obox[2], :]
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            image = image[None, :, :, :]
            print(image.shape)
            self.obox = None
        return image
