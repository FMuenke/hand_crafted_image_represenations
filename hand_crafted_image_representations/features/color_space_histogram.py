import numpy as np

from hand_crafted_image_representations.data_structure.image_handler import ImageHandler
from hand_crafted_image_representations.machine_learning.feature_map import FeatureMap


class ColorSpaceHistogram:
    def __init__(self, color_space, resolution=64):
        self.color_space = color_space
        self.resolution = resolution

    def _prepare_image(self, image):
        img_h = ImageHandler(image)
        return img_h.prepare_image_for_processing(self.color_space)

    def _collect_histograms(self, list_of_targets, key_points):
        dc_sets = []
        for target in list_of_targets:
            f_map = FeatureMap(target)
            dc = f_map.to_descriptors_with_histogram(key_points, resolution=self.resolution)
            if dc is not None:
                dc_sets.append(dc)
        return dc_sets

    def compute(self, image, key_points):
        list_of_targets = self._prepare_image(image)
        dc_sets = self._collect_histograms(list_of_targets, key_points)
        if len(dc_sets) == 0:
            return None
        elif len(dc_sets) == 1:
            return dc_sets[0]
        else:
            return np.concatenate(dc_sets, axis=1)
