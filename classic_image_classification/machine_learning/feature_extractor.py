import numpy as np
from tqdm import tqdm

from classic_image_classification.machine_learning.descriptor_set import DescriptorSet
from classic_image_classification.machine_learning.key_point_set import KeyPointSet
from classic_image_classification.data_structure.image_handler import ImageHandler


class FeatureExtractor:
    def __init__(self,
                 features_to_use,
                 normalize=True,
                 sampling_method="dense",
                 sampling_steps=20,
                 sampling_window=20,
                 image_height=None,
                 image_width=None,
                 ):
        if type(features_to_use) is not list():
            features_to_use = [features_to_use]
        self.features_to_use = features_to_use
        self.normalize = normalize
        self.sampling_method = sampling_method
        self.sampling_steps = sampling_steps
        self.sampling_window = sampling_window

        self.img_height = image_height
        self.img_width = image_width

        self.min_height = 16
        self.min_width = 16

    def resize(self, image, image_size):
        img_h = ImageHandler(image)
        if None in image_size:
            height, width = image.shape[:2]
            if height < self.min_height:
                height = self.min_height

            if width < self.min_width:
                width = self.min_width
            return img_h.resize(height=height, width=width)
        return img_h.resize(height=image_size[0], width=image_size[1])

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

    def extract_trainings_data(self, tags):
        x = []
        y = []
        print("[INFO] Extracting Features: {} for Tags".format(self.features_to_use))
        for tag_id in tqdm(tags):
            tag_data = tags[tag_id].load_data()
            x_tag = self.extract_x(tag_data)
            y_tag = tags[tag_id].load_y()

            x.append(x_tag)
            y.append(y_tag)
        assert x is not None, "No Feature was activated"
        return x, y

    def extract_x(self, image):
        kp_set = KeyPointSet(self.sampling_method, self.sampling_steps, self.sampling_window)
        image = self.resize(image, [self.img_height, self.img_width])
        x = []
        for feature in self.features_to_use:
            dc_set = DescriptorSet(feature)
            dc_x = dc_set.compute(image, kp_set)
            if dc_x is not None:
                x.append(dc_x)
        return self._output_list(x, axis=1)