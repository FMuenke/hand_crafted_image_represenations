import numpy as np
from features.local_binary_pattern import LocalBinaryPattern
from features.leung_malik import LeungMalik
from features.color_space_histogram import ColorSpaceHistogram
from features.open_cv_descriptor import OpenCvDescriptor
from features.histogram_of_oriented_gradients import HistogramOfOrientedGradients
from features.gray_co_matrix import GrayCoMatrix
from features.haar import Haar


class DescriptorSet:
    def __init__(self, descriptor_type):
        self.descriptor_type = descriptor_type

    def _output_list(self, input_list, axis):
        if input_list is not None:
            if len(input_list) == 0:
                return None
            elif len(input_list) == 1:
                return input_list[0]
            else:
                return np.concatenate(input_list, axis=axis)
        else:
            return None

    def compute(self, image, key_point_set):
        if "lbp" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            lbp = LocalBinaryPattern(color_space=color_space)
            return lbp.compute(image, key_point_set.get_key_points(image))
        if "lm" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            lm = LeungMalik(color_space)
            return lm.compute(image, key_point_set.get_key_points(image))
        if "histogram" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            c_s_hist = ColorSpaceHistogram(color_space=color_space)
            return c_s_hist.compute(image, key_point_set.get_key_points(image))
        if "hog" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            hog = HistogramOfOrientedGradients(color_space=color_space)
            return hog.compute(image, key_point_set.get_key_points(image))
        if "glcm" == self.descriptor_type:
            gclm = GrayCoMatrix()
            return gclm.compute(image, key_point_set.get_key_points(image))
        if "haar" == self.descriptor_type:
            haar = Haar()
            return haar.compute(image, key_point_set.get_key_points(image))
        if "kaze" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
            return cv.compute(image, key_point_set.get_open_cv_key_points(image))
        if "akaze" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
            return cv.compute(image, key_point_set.get_open_cv_key_points(image))
        if "brisk" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
            return cv.compute(image, key_point_set.get_open_cv_key_points(image))
        if "orb" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
            return cv.compute(image, key_point_set.get_open_cv_key_points(image))
        if "sift" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
            return cv.compute(image, key_point_set.get_open_cv_key_points(image))

        raise ValueError("Unknown Descriptor Type {}".format(self.descriptor_type))

