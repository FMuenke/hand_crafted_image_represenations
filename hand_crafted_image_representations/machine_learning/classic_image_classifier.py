from hand_crafted_image_representations.utils.utils import check_n_make_dir
import os
import numpy as np
import logging
from hand_crafted_image_representations.machine_learning.feature_extractor import FeatureExtractor
from hand_crafted_image_representations.machine_learning.aggregator import Aggregator
from hand_crafted_image_representations.machine_learning.classifier import Classifier

from hand_crafted_image_representations.data_structure.data_set import DataSet

from hand_crafted_image_representations.utils.utils import save_dict, load_dict
from hand_crafted_image_representations.utils.statistic_utils import init_result_dict, show_results, save_results

from tqdm import tqdm

from hand_crafted_image_representations.utils.statistic_utils import plot_roc


class ClassicImageClassifier:
    def __init__(self, opt=None, class_mapping=None):
        self.opt = opt

        self.class_mapping = class_mapping
        self.class_mapping_inv = None

        self.feature_extractor = None
        self.aggregator = None
        self.classifier = None

    def new(self):
        if "sampling_step" not in self.opt:
            self.opt["sampling_step"] = 20
        if "sampling_window" not in self.opt:
            self.opt["sampling_window"] = 20

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

        logging.info("Machine-Learning-Pipeline was saved to: {}".format(model_path))

    def extract(self, tag_set):
        return self.feature_extractor.extract_trainings_data(tag_set)

    def build_aggregator(self, descriptors):
        logging.info("Feature Aggregator was added to the machine learning pipeline")
        self.aggregator.fit(descriptors)

    def aggregate(self, x):
        return self.aggregator.transform(x)

    def fit(self, data_path, tag_type, load_all=False, report_path=None):
        self.new()
        ds = DataSet(data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        assert len(tags) != 0, "No Labels were found! Abort..."

        x_train, y_train = self.feature_extractor.extract_trainings_data(tags)
        x_train = self.aggregator.fit_transform(x_train)
        x_train = np.concatenate(x_train, axis=0)

        self.classifier.fit(x_train, y_train)
        logging.info(self.classifier)
        score = self.classifier.evaluate(x_train, y_train, save_path=report_path)
        return score

    def predict_image(self, image, get_confidence=False):
        x = self.feature_extractor.extract_x(image)
        if self.aggregator.is_fitted():
            x = self.aggregator.transform([x])
            x = x[0]
        return self.classifier.predict(x, get_confidence)

    def evaluate(self, data_path, tag_type, load_all=False, report_path=None):
        ds = DataSet(data_set_dir=data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        assert len(tags) != 0, "No Labels were found! Abort..."

        if report_path is not None:
            check_n_make_dir(report_path, clean=True)

        logging.info("Running Inference ...")
        result_dict = init_result_dict(self.class_mapping)
        y = []
        predictions = []
        confidences = []
        for tag in tqdm(tags):
            y_pred, conf = self.predict_image(tag.load_data(), get_confidence=True)
            if report_path is not None:
                tag.write_prediction(y_pred, report_path)
            y.append(tag.load_y())
            predictions.append(y_pred[0])
            confidences.append(conf)
            result_dict = tag.evaluate_prediction(y_pred, result_dict)

        show_results(result_dict)

        if report_path is not None:
            plot_roc(y, predictions, confidences, self.class_mapping, report_path)
            save_results(report_path, "image_classifier", result_dict)
