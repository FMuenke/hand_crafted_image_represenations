from classic_image_classification.utils.utils import check_n_make_dir
import os
import numpy as np
import logging
from classic_image_classification.machine_learning.feature_extractor import FeatureExtractor
from classic_image_classification.machine_learning.aggregator import Aggregator
from classic_image_classification.machine_learning.classifier_handler import ClassifierHandler
from classic_image_classification.machine_learning.data_sampler import DataSampler

from classic_image_classification.data_structure.data_set import DataSet

from classic_image_classification.utils.utils import save_dict, load_dict


class ClassicImageClassifier:
    def __init__(self, opt=None, class_mapping=None):

        self.opt = opt

        self.class_mapping = class_mapping
        self.class_mapping_inv = None

        self.aggregator = None
        self.feature_extractor = None
        self.classifier = None

    def new(self):
        self.feature_extractor = FeatureExtractor(
            features_to_use=self.opt["feature"],
            image_height=self.opt["image_size"]["height"],
            image_width=self.opt["image_size"]["width"],
            sampling_method=self.opt["sampling_method"],
            sampling_steps=self.opt["sampling_step"],
            sampling_window=self.opt["sampling_window"]
        )

        self.aggregator = Aggregator(self.opt)
        self.classifier = ClassifierHandler(self.opt, self.class_mapping)

    def load(self, model_path):
        path_to_opt = os.path.join(model_path, "classifier_opt.json")
        self.opt = load_dict(path_to_opt)
        self.new()
        path_to_class_mapping = os.path.join(model_path, "class_mapping.json")
        self.class_mapping = load_dict(path_to_class_mapping)
        self.class_mapping_inv = {v: k for k, v in self.class_mapping.items()}

        self.classifier.load(model_path)
        self.aggregator.load(model_path)

    def save(self, model_path):
        check_n_make_dir(model_path, clean=False)
        path_to_opt = os.path.join(model_path, "classifier_opt.json")
        save_dict(self.opt, path_to_opt)
        path_to_class_mapping = os.path.join(model_path, "class_mapping.json")
        save_dict(self.class_mapping, path_to_class_mapping)

        self.aggregator.save(model_path)
        self.classifier.save(model_path)

        print("machine_learning-Pipeline was saved to: {}".format(model_path))

    def extract(self, tag_set):
        return self.feature_extractor.extract_trainings_data(tag_set)

    def build_aggregator(self, descriptors):
        print("Bag of Words was added to the machine learning pipeline")
        self.aggregator.fit(descriptors)

    def aggregate(self, x):
        return self.aggregator.transform(x)

    def fit(self, data_path, tag_type, load_all=False, report_path=None):
        self.new()
        ds = DataSet(data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        ds.load_data()
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        assert len(tags) != 0, "No Labels were found! Abort..."

        x, y = self.feature_extractor.extract_trainings_data(tags)
        x = self.aggregator.fit_transform(x)

        x = np.concatenate(x, axis=0)

        data_sampler = DataSampler(self.class_mapping, x=x, y=y, mode=self.opt["data_split_mode"])
        x_train, x_test, y_train, y_test = data_sampler.train_test_split(percentage=self.opt["data_split_ratio"])

        if "param_grid" in self.opt:
            self.classifier.fit_inc_hyper_parameter(x_train, y_train, self.opt["param_grid"], n_iter=100)
        else:
            self.classifier.fit(x_train, y_train)
        logging.info(self.classifier)
        score = self.classifier.evaluate(x_test, y_test, save_path=report_path)
        return score

    def predict(self, tag, get_confidence=False):
        if "roi" in self._pipeline_opt["image_size"]:
            tag.set_roi(self._pipeline_opt["image_size"]["roi"])
        X = self.feature_extractor.extract_x(tag.load_data())
        if self.aggregator.is_fitted():
            X = self.aggregator.transform([X])
            X = X[0]
        return self.classifier.predict(X, get_confidence)
