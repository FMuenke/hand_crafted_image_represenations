import numpy as np
import cv2
from skimage.transform import integral_image

from handcrafted_image_representations.data_structure.matrix_handler import MatrixHandler


class ImageHandler:
    def __init__(self, image):
        self.image = image
        self.transformed_img = image

    def prepare_image_for_processing(self, color_space):
        if color_space == "gray":
            return [self.gray()]
        elif color_space == "hsv":
            hsv = self.hsv()
            return [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]
        elif color_space == "BGR":
            BGR = self.BGR()
            return [BGR[:, :, 0], BGR[:, :, 1], BGR[:, :, 2]]
        elif color_space == "RGB":
            RGB = self.RGB()
            return [RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2]]
        elif color_space == "opponent":
            opponent = self.opponent()
            return [opponent[:, :, 0], opponent[:, :, 1], opponent[:, :, 2]]
        elif color_space == "rgb":
            rgb = self.rgb()
            return [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
        else:
            raise ValueError("Color Space {} unknown.".format_map(color_space))

    def BGR(self):
        return self.image

    def RGB(self):
        B = self.image[:, :, 0]
        G = self.image[:, :, 1]
        R = self.image[:, :, 2]
        R = np.expand_dims(R, axis=2)
        B = np.expand_dims(B, axis=2)
        G = np.expand_dims(G, axis=2)
        return np.concatenate([R, G, B], axis=2)

    def gray(self):
        image = np.copy(self.image)
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def integral(self):
        return integral_image(self.gray())

    def hsv(self):
        image = np.copy(self.image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def opponent(self):
        B = self.image[:, :, 0]
        G = self.image[:, :, 1]
        R = self.image[:, :, 2]
        O1 = np.divide((R - G), np.sqrt(2))
        O2 = np.divide((R + G - np.multiply(B, 2)), np.sqrt(6))
        O3 = np.divide((R + G + B), np.sqrt(3))
        O1 = np.expand_dims(O1, axis=2)
        O2 = np.expand_dims(O2, axis=2)
        O3 = np.expand_dims(O3, axis=2)
        return np.concatenate([O1, O2, O3], axis=2)

    def rgb(self):
        eps = 1e-6
        self.image = self.image.astype(np.float64)
        B = self.image[:, :, 0] + eps
        G = self.image[:, :, 1] + eps
        R = self.image[:, :, 2] + eps
        r = np.divide(R, (R + G + B))
        g = np.divide(G, (R + G + B))
        b = np.divide(B, (R + G + B))
        r = 255 * np.expand_dims(r, axis=2)
        b = 255 * np.expand_dims(b, axis=2)
        g = 255 * np.expand_dims(g, axis=2)
        return np.concatenate([r, b, g], axis=2).astype(np.int32)

    def normalize(self):
        image = np.copy(self.image)
        mat = MatrixHandler(image)
        img_norm = 255 * mat.normalize()
        return img_norm.astype(np.uint8)

    def resize(self, height, width):
        print(height, width, self.image.shape[:2])
        if height is None and width is None:
            return self.image
        o_height, o_width = self.image.shape[:2]
        if height is None:
            height = int(o_height * (width / o_width))
        if width is None:
            width = int(o_width * (height / o_height))
        print(height, width)
        return cv2.resize(
            self.image,
            (int(width), int(height)),
            interpolation=cv2.INTER_CUBIC
        )
    
    def _scale_image_to_octave(self, octave):
        image = np.copy(self.image)
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return image
        elif octave == 0:
            return image
        else:
            oct_width = int(width / np.power(2, octave))
            oct_height = int(height / np.power(2, octave))
            return self.resize(height=oct_height, width=oct_width)

    def overlay(self, mask):
        height, width = self.image.shape[:2]
        mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        return cv2.addWeighted(self.image.astype(np.uint8), 0.5, mask.astype(np.uint8), 0.5, 0)

    def write(self, write_path: str):
        if not write_path.endswith(".jpg"):
            write_path += ".jpg"
        cv2.imwrite(write_path, self.transformed_img)

    def attach(self, image):
        height, width = self.image.shape[:2]
        mask = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        border = 255 * np.ones((height, 10, 3))
        self.transformed_img = np.concatenate([self.transformed_img, border, mask], axis=1)
