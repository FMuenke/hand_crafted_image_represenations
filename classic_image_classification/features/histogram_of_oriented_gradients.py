import cv2
import numpy as np


from classic_image_classification.data_structure.image_handler import ImageHandler
from classic_image_classification.machine_learning.feature_map import FeatureMap


class HistogramOfOrientedGradients:
    def __init__(self, color_space="gray", orientations=32, norm_option="L2HYS"):
        self.color_space = color_space
        self.orientations = orientations

        if "NORM" in norm_option:
            self.norm_orientation = True
        else:
            self.norm_orientation = False
        self.norm_option = norm_option.replace("NORM", "")

    def build_f_maps(self, image):
        gx = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(np.copy(image.astype(np.uint8)), cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        ang_sorted = ang / (2 * np.pi) * self.orientations
        ang_sorted = ang_sorted.astype(np.int)

        gradient_maps = []
        for ori_idx in range(self.orientations):
            grad_map = np.zeros(mag.shape)
            grad_map[ang_sorted == ori_idx] = mag[ang_sorted == ori_idx]
            gradient_maps.append(grad_map)

        return np.stack(gradient_maps, axis=2)

    def norm_dc_set(self, dc, norm_option="L2HYS"):
        eps = 1e-6
        if norm_option == "L2HYS":
            dc_norm = self.norm_dc_set(dc, norm_option="L2")
            dc_norm = np.clip(dc_norm, 0, 0.2)
            dc_norm = self.norm_dc_set(dc_norm, norm_option="L2")
            return dc_norm
        if norm_option == "L2":
            num_features = dc.shape[1]
            norm_factor = np.sqrt(np.sum(np.power(dc, 2), axis=1)) + eps
            norm_factor = np.reshape(norm_factor, (-1, 1))
            norm_factor = np.repeat(norm_factor, num_features, axis=1)
            dc_norm = np.divide(dc, norm_factor)
            return dc_norm
        return dc

    def _compute(self, channels):
        f_maps = []
        for c in channels:
            f_map = self.build_f_maps(c)
            f_maps.append(f_map)
        return f_maps

    def _collect_descriptors(self, list_of_targets, key_points):
        dc_sets = []
        for target in list_of_targets:
            f_map = FeatureMap(target)
            dc = f_map.to_descriptor_with_pooling(key_points,
                                                  pooling_mode="sum",
                                                  roll_to_max_first=self.norm_orientation)
            dc = self.norm_dc_set(dc, norm_option=self.norm_option)
            if dc is not None:
                dc_sets.append(dc)
        return dc_sets

    def compute(self, image, key_points):
        img_h = ImageHandler(image)
        channels = img_h.prepare_image_for_processing(self.color_space)
        f_maps = self._compute(channels)
        dc_sets = self._collect_descriptors(f_maps, key_points)
        if len(dc_sets) == 0:
            return None
        elif len(dc_sets) == 1:
            return dc_sets[0]
        else:
            return np.concatenate(dc_sets, axis=1)


