from classic_image_classification.features.local_binary_pattern import LocalBinaryPattern
from classic_image_classification.features.leung_malik import LeungMalik
from classic_image_classification.features.color_space_histogram import ColorSpaceHistogram
from classic_image_classification.features.open_cv_descriptor import OpenCvDescriptor
from classic_image_classification.features.histogram_of_oriented_gradients import HistogramOfOrientedGradients


class DescriptorSet:
    def __init__(self, descriptor_type):
        self.descriptor_type = descriptor_type

    def compute(self, image, key_point_set):
        if "lbp" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            if "+" in descriptor_type:
                descriptor_type, num_points, radius = descriptor_type.split("+")
            else:
                num_points, radius = 24, 7
            lbp = LocalBinaryPattern(color_space=color_space,
                                     num_points=int(num_points),
                                     radius=int(radius))
            return lbp.compute(image, key_point_set.get_key_points(image))
        if "lm" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            lm = LeungMalik(color_space)
            return lm.compute(image, key_point_set.get_key_points(image))
        if "histogram" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            if "+" in descriptor_type:
                descriptor_type, num_bins = descriptor_type.split("+")
            else:
                num_bins = 64
            c_s_hist = ColorSpaceHistogram(color_space=color_space,
                                           resolution=int(num_bins))
            return c_s_hist.compute(image, key_point_set.get_key_points(image))
        if "hog" in self.descriptor_type:
            color_space, descriptor_type = self.descriptor_type.split("-")
            if "+" in descriptor_type:
                descriptor_type, cfg, norm_option = descriptor_type.split("+")
            else:
                cfg, norm_option = "SMALL", "L2HYS"
            hog = HistogramOfOrientedGradients(color_space=color_space,
                                               config=cfg,
                                               norm_option=norm_option)
            return hog.compute(image, key_point_set.get_key_points(image))

        for feat in ["kaze", "akaze", "sift", "brisk", "orb"]:
            if feat in self.descriptor_type:
                color_space, descriptor_type = self.descriptor_type.split("-")
                cv = OpenCvDescriptor(descriptor_type=descriptor_type, color_space=color_space)
                return cv.compute(image, key_point_set.get_open_cv_key_points(image))

        raise ValueError("Unknown Descriptor Type {}".format(self.descriptor_type))
