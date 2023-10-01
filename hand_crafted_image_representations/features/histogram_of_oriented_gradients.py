import numpy as np
from skimage import feature

from hand_crafted_image_representations.data_structure.image_handler import ImageHandler
from hand_crafted_image_representations.data_structure.matrix_handler import MatrixHandler


class HistogramOfOrientedGradients:
    def __init__(self, color_space="gray", config="SMALL", norm_option="L2HYS"):
        self.color_space = color_space
        self.config = config
        norm_mapping = {
            "L1": "L1", "L1SQRT": "L1-sqrt",
            "L2HYS": "L2-Hys", "L2": "L2",
        }
        self.norm_option = norm_mapping[norm_option]
        self.orientations = 9

        if config == "TINY":
            self.pixels_per_cell = (4, 4)
            self.cells_per_block = (1, 1)
        if config == "SMALL":
            self.pixels_per_cell = (8, 8)
            self.cells_per_block = (2, 2)
        if config == "BIG":
            self.pixels_per_cell = (16, 16)
            self.cells_per_block = (3, 3)
        if config == "HUGE":
            self.pixels_per_cell = (32, 32)
            self.cells_per_block = (4, 4)

    def _collect_descriptors(self, target_list, key_points):
        dc_sets = []
        for x, y, roi_size in key_points:
            dc_targets = []
            for target in target_list:
                mat = MatrixHandler(target)
                roi = mat.cut_roi([x, y], roi_size)
                dc = feature.hog(
                    image=roi,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    orientations=self.orientations,
                    block_norm=self.norm_option
                )
                dc_targets.append(dc)
            dc_targets = np.concatenate(dc_targets)
            dc_targets = np.reshape(dc_targets, (1, -1))
            dc_sets.append(dc_targets)
        return dc_sets

    def compute(self, image, key_points):
        if len(key_points) == 0:
            return None
        img_h = ImageHandler(image)
        channels = img_h.prepare_image_for_processing(self.color_space)
        dc_sets = self._collect_descriptors(channels, key_points)
        if len(dc_sets) == 0:
            return None
        elif len(dc_sets) == 1:
            return dc_sets[0]
        else:
            return np.concatenate(dc_sets, axis=1)
