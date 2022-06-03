from classic_image_classification.utils.utils import check_n_make_dir
import os
import numpy as np
import logging
from classic_image_classification.machine_learning.feature_extractor import FeatureExtractor
from classic_image_classification.machine_learning.aggregator import Aggregator
from classic_image_classification.machine_learning.classifier import Classifier
from classic_image_classification.machine_learning.data_sampler import DataSampler
from classic_image_classification.utils.data_split import split_tags

from classic_image_classification.data_structure.data_set import DataSet

from classic_image_classification.utils.utils import save_dict, load_dict
from classic_image_classification.utils.statistic_utils import init_result_dict, show_results, save_results

from tqdm import tqdm


class ClassicImageClassifier:
    def __init__(self, opt=None, class_mapping=None):
        self.opt = opt

        self.class_mapping = class_mapping
        self.class_mapping_inv = None

        self.feature_extractor = None
        self.aggregator = None
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
        self.classifier = Classifier(self.opt, self.class_mapping)

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
        print("Feature Aggregator was added to the machine learning pipeline")
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

        train_tags, test_tags = split_tags(tags)
        x_train, y_train = self.feature_extractor.extract_trainings_data(train_tags)
        x_test, y_test = self.feature_extractor.extract_trainings_data(train_tags)
        x_train = self.aggregator.fit_transform(x_train)
        x_test = self.aggregator.transform(x_test)

        x_train = np.concatenate(x_train, axis=0)
        x_test = np.concatenate(x_test, axis=0)

        self.classifier.fit(x_train, y_train)
        logging.info(self.classifier)
        score = self.classifier.evaluate(x_test, y_test, save_path=report_path)
        return score

    def predict_image(self, image, get_confidence=False):
        X = self.feature_extractor.extract_x(image)
        if self.aggregator.is_fitted():
            X = self.aggregator.transform([X])
            X = X[0]
        return self.classifier.predict(X, get_confidence)

    def evaluate(self, data_path, tag_type, load_all=False, report_path=None):
        ds = DataSet(data_set_dir=data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        ds.load_data()
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        assert len(tags) != 0, "No Labels were found! Abort..."

        logging.info("Running Inference ...")
        result_dict = init_result_dict(self.class_mapping)
        for tag_id in tqdm(tags):
            tag = tags[tag_id]
            y_pred = self.predict_image(tag.load_data())
            if report_path is not None:
                tag.write_prediction(y_pred, report_path)
            result_dict = tag.evaluate_prediction(y_pred, result_dict)

        show_results(result_dict)
        if report_path is not None:
            save_results(report_path, "image_classifier", result_dict)
