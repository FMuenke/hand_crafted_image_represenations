import joblib
from utils.utils import check_n_make_dir
import os
import numpy as np

from machine_learning.feature_extractor import FeatureExtractor
from machine_learning.aggregator_handler import AggregatorHandler
from machine_learning.classifier_handler import ClassifierHandler
from machine_learning.data_sampler import DataSampler

from utils.utils import save_dict, load_dict


class ClassicImageClassifier:
    def __init__(self, model_path, pipeline_opt=None, class_mapping=None):
        self.model_path = model_path
        self.pipeline_name = os.path.basename(model_path)

        self._pipeline_opt = pipeline_opt

        self.class_mapping = class_mapping
        self.class_mapping_inv = None

        self.aggregator = None
        self.feature_extractor = None
        self.classifier = None

        self.path_to_opt = os.path.join(self.model_path, "opt.json")
        self.path_to_class_mapping = os.path.join(self.model_path, "class_mapping.json")
        self.aggregator_path = os.path.join(self.model_path, "aggregator.pkl")
        self.feature_extractor_path = os.path.join(self.model_path, "feature_extractor.pkl")
        self.classifier_path = os.path.join(self.model_path, "classifier.pkl")

    def new(self):
        self.feature_extractor = FeatureExtractor(features_to_use=self._pipeline_opt["feature"],
                                                  image_height=self._pipeline_opt["image_size"]["height"],
                                                  image_width=self._pipeline_opt["image_size"]["width"],
                                                  sampling_method=self._pipeline_opt["sampling_method"],
                                                  sampling_steps=self._pipeline_opt["sampling_step"],
                                                  sampling_window=self._pipeline_opt["sampling_window"],
                                                  padding=self._pipeline_opt["image_size"]["padding"])

        self.aggregator = AggregatorHandler(self.model_path, self._pipeline_opt)
        self.classifier = ClassifierHandler(self.model_path, self._pipeline_opt, self.class_mapping)

    def extract(self, tag_set):
        if "roi" in self._pipeline_opt["image_size"]:
            print("Setting region of Interest...")
            for _, t in tag_set.items():
                t.set_roi(self._pipeline_opt["image_size"]["roi"])
        return self.feature_extractor.build_x_y(tag_set)

    def build_aggregator(self, descriptors):
        print("Bag of Words was added to the machine learning pipeline")
        self.aggregator.fit(descriptors)

    def aggregate(self, x):
        return self.aggregator.transform(x)

    def fit(self, x, y, percentage=0.2):
        x = np.concatenate(x, axis=0)
        try:
            y = np.concatenate(y, axis=0)
        except:
            y = np.array(y)

        data_sampler = DataSampler(self.class_mapping, x=x, y=y, mode=self._pipeline_opt["data_split_mode"])
        x_train, x_test, y_train, y_test = data_sampler.train_test_split(percentage=percentage)

        self.classifier.new_classifier()
        if "param_grid" in self._pipeline_opt["classifier_opt"]:
            self.classifier.fit_inc_hyper_parameter(x_train, y_train, self._pipeline_opt["classifier_opt"]["param_grid"], n_iter=100)
        else:
            self.classifier.fit(x_train, y_train)
        print(self.classifier)
        score = self.classifier.evaluate(x_test, y_test, save_path=os.path.join(self.model_path, "eval_report.txt"))
        return score

    def load(self):
        self.class_mapping = load_dict(self.path_to_class_mapping)
        self.class_mapping_inv = {v: k for k, v in self.class_mapping.items()}
        self.feature_extractor = joblib.load(self.feature_extractor_path)
        self.classifier = joblib.load(self.classifier_path)
        self.aggregator = joblib.load(self.aggregator_path)
        self._pipeline_opt = load_dict(self.path_to_opt)

    def save(self):
        check_n_make_dir(self.model_path, clean=False)
        print("machine_learning-Pipeline was saved to: {}".format(self.model_path))
        save_dict(self.class_mapping, self.path_to_class_mapping)
        save_dict(self._pipeline_opt, self.path_to_opt)
        joblib.dump(self.feature_extractor, self.feature_extractor_path)
        joblib.dump(self.classifier, self.classifier_path)
        joblib.dump(self.aggregator, self.aggregator_path)

    def predict(self, tag, get_confidence=False):
        if "roi" in self._pipeline_opt["image_size"]:
            tag.set_roi(self._pipeline_opt["image_size"]["roi"])
        X = self.feature_extractor.build_x(tag.load_data())
        if self.aggregator.is_fitted():
            X = self.aggregator.transform([X])
            X = X[0]
        return self.classifier.predict(X, get_confidence)
