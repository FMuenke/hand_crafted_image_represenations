import cv2
import numpy as np

from handcrafted_image_representations.data_structure.image_handler import ImageHandler


class OpenCvDescriptor:
    def __init__(self, descriptor_type, color_space):
        self.descriptor_type = descriptor_type
        self.color_space = color_space

    def _build_open_cv_descriptor(self):
        open_cv_key_point_detectors = {
            "orb": cv2.ORB_create(),
            "akaze": cv2.AKAZE_create(),
            "kaze": cv2.KAZE_create(),
            "brisk": cv2.BRISK_create(),
            "sift": cv2.SIFT_create(),
        }
        return open_cv_key_point_detectors

    def _prepare_image(self, image):
        img_h = ImageHandler(image)
        return img_h.prepare_image_for_processing(self.color_space)

    def _apply_open_cv_descriptor(self, list_of_targets, open_cv_key_points):
        open_cv_descriptors = self._build_open_cv_descriptor()
        dc_sets = []
        for target in list_of_targets:
            target = target.astype(np.uint8)
            kp, desc = open_cv_descriptors[self.descriptor_type].compute(target, open_cv_key_points)
            dc_sets.append(desc)
        return dc_sets

    def compute(self, image, open_cv_key_points):
        list_of_targets = self._prepare_image(image)
        dc_sets = self._apply_open_cv_descriptor(list_of_targets, open_cv_key_points)

        if len(dc_sets) == 0:
            return None
        elif len(dc_sets) == 1:
            return dc_sets[0]
        else:
            return np.concatenate(dc_sets, axis=1)
