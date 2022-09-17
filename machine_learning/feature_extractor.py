import numpy as np
from tqdm import tqdm

from machine_learning.descriptor_set import DescriptorSet
from machine_learning.key_point_set import KeyPointSet

from pre.preprocessor import Preprocessor


class FeatureExtractor:
    def __init__(self,
                 features_to_use,
                 normalize=True,
                 sampling_method="dense",
                 sampling_steps=20,
                 sampling_window=20,
                 image_height=None,
                 image_width=None,
                 resize_option="standard",
                 ):

        self.features_to_use = features_to_use
        self.normalize = normalize
        self.sampling_method = sampling_method
        self.sampling_steps = sampling_steps
        self.sampling_window = sampling_window
        self.resize_option = resize_option

        self.img_height = image_height
        self.img_width = image_width

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

    def _build_x_bag_of_words(self, image):
        kp_set = KeyPointSet(self.sampling_method, self.sampling_steps, self.sampling_window)
        p = Preprocessor([self.img_height, self.img_width], normalize=False, resize_option=self.resize_option)
        image = p.apply(image)
        x = []
        for feature in self.features_to_use:
            dc_set = DescriptorSet(feature)
            dc_x = dc_set.compute(image, kp_set)
            if dc_x is not None:
                x.append(dc_x)
        return self._output_list(x, axis=1)

    def build_x_y(self, tags):
        x = []
        y = []
        print("Extracting Features: {} for Tags".format(self.features_to_use))
        for tag_id in tqdm(tags):
            tag_data = tags[tag_id].load_data()
            x_tag = self._build_x_bag_of_words(tag_data)
            y_tag = tags[tag_id].load_y()

            x.append(x_tag)
            y.append(y_tag)
        assert x is not None, "No Feature was activated"
        return x, y

    def build_x(self, tag_data):
        return self._build_x_bag_of_words(tag_data)
