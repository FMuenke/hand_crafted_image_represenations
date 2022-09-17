import os
from classic_image_classification.utils.utils import save_dict, load_dict, check_n_make_dir

from classic_image_classification.machine_learning.feature_extractor import FeatureExtractor
from classic_image_classification.machine_learning.aggregator import Aggregator

from classic_image_classification.data_structure.data_set import DataSet


class ImageEmbedding:
    def __init__(self, opt):
        self.opt = opt
        self.feature_extractor = None
        self.feature_aggregator = None

    def new(self):
        self.feature_extractor = FeatureExtractor(
            features_to_use=self.opt["feature"],
            image_height=self.opt["image_size"]["height"],
            image_width=self.opt["image_size"]["width"],
            sampling_method=self.opt["sampling_method"],
            sampling_steps=self.opt["sampling_step"],
            sampling_window=self.opt["sampling_window"]
        )

        self.feature_aggregator = Aggregator(self.opt)

    def fit(self, data_path, tag_type, classes_to_consider="all"):
        self.new()
        ds = DataSet(data_path, tag_type=tag_type)
        ds.load_data()
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        x, _ = self.feature_extractor.extract_trainings_data(tags)
        self.feature_aggregator.fit(x)

    def transform(self, image):
        x = self.feature_extractor.extract_x(image)
        x = self.feature_aggregator.transform(x)
        return x

    def fit_transform(self, data_path, tag_type, classes_to_consider="all"):
        self.new()
        ds = DataSet(data_path, tag_type=tag_type)
        ds.load_data()
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        x, _ = self.feature_extractor.extract_trainings_data(tags)
        return self.feature_aggregator.fit_transform(x)

    def save(self, path):
        check_n_make_dir(path)
        opt_file = os.path.join(path, "opt.json")
        save_dict(self.opt, opt_file)
        self.feature_aggregator.save(path)

    def load(self, path):
        opt_file = os.path.join(path, "opt.json")
        self.opt = load_dict(opt_file)
        self.new()
        self.feature_aggregator.load(path)
