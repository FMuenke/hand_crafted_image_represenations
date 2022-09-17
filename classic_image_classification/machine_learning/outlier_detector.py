import copy
import numpy as np
from sklearn.model_selection import ParameterGrid
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification import machine_learning as ml

import os
import shutil
from classic_image_classification.utils.utils import check_n_make_dir

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import roc_auc_score


class OutlierByDistanceMultiClass:
    def __init__(self):
        self.remover = None

    def fit(self, x, y):
        unique_y = np.unique(y)
        self.remover = {}
        for uy in unique_y:
            self.remover[uy] = OutlierByDistance()
            ux = x[y == uy, :]
            self.remover[uy].fit(ux)

    def score_sample(self, x):
        sep_dist = []
        for uy in self.remover:
            u_dist = self.remover[uy].compute_distance(x)
            u_dist = np.expand_dims(u_dist, axis=1)
            sep_dist.append(u_dist)

        sep_dist = np.concatenate(sep_dist, axis=1)
        min_dist = np.min(sep_dist, axis=1)
        return min_dist * -1


class OutlierByDistance:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.max_distance = None
        self.min_distance = None

    def compute_distance(self, x):
        distance = np.square((x - self.x_mean) / self.x_std)
        return np.sqrt(np.sum(distance, axis=1))

    def fit(self, x, y=None):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        self.x_std[self.x_std == 0.0] = 1e-9

        dist = self.compute_distance(x)
        self.max_distance = np.max(dist)
        self.min_distance = np.min(dist)

    def score_sample(self, x):
        return -1 * self.compute_distance(x)

    def predict(self, x):
        distance = self.compute_distance(x)
        prediction = np.zeros(distance.shape)
        prediction[distance > self.max_distance] = -1
        prediction[distance <= self.max_distance] = 1
        return prediction


class OutlierDetector:
    def __init__(self, opt, class_mapping):
        self.opt = opt
        self.class_mapping = class_mapping

        self.aggregator_opt = ["aggregator", "complexity"]

        self.feature_extractor = None
        self.aggregator_list = None
        self.remover_list = None
        self.remover = None

        self.final_remover = None

    def new(self):
        self.feature_extractor = ml.FeatureExtractor(
            features_to_use=self.opt["feature"],
            image_height=self.opt["image_size"]["height"],
            image_width=self.opt["image_size"]["width"],
            sampling_method=self.opt["sampling_method"],
            sampling_steps=self.opt["sampling_step"],
            sampling_window=self.opt["sampling_window"]
        )

        for k in self.aggregator_opt:
            if k not in self.opt:
                continue
            if type(self.opt[k]) is not list:
                self.opt[k] = [self.opt[k]]

        aggregator_opt_list = list(ParameterGrid({k: self.opt[k] for k in self.aggregator_opt if k in self.opt}))

        self.aggregator_list = [ml.Aggregator(opt) for opt in aggregator_opt_list]
        self.remover_list = [
            IsolationForest(n_jobs=-1),
            LocalOutlierFactor(novelty=True, n_jobs=-1),
            OutlierByDistance(),
        ]
        self.remover = OutlierByDistance()

    def fit(self, model_folder, data_path_known, data_path_test, tag_type, report_path=None):
        self.new()
        best_score = 0
        best_candidate = None

        ds = DataSet(data_path_known, tag_type, self.class_mapping)
        ds.load_data()
        ds_test = DataSet(data_path_test, tag_type, self.class_mapping)
        ds_test.load_data()
        tags = ds.get_tags(self.class_mapping)
        tags_test = ds_test.get_tags(classes_to_consider="all")

        x, y = self.feature_extractor.extract_trainings_data(tags)
        x_test, y_test = self.feature_extractor.extract_trainings_data(tags_test)
        y_test = np.array(y_test)
        y_test[y_test != -1] = 1

        for aggregator in self.aggregator_list:
            x_transformed = aggregator.fit_transform(x)
            x_transformed_test = aggregator.transform(x_test)
            x_transformed = np.concatenate(x_transformed, axis=0)
            x_transformed_test = np.concatenate(x_transformed_test, axis=0)

            self.remover.fit(x_transformed, y)
            y_rm = self.remover.score_sample(x_transformed_test)
            score = roc_auc_score(y_test, y_rm)
            if score > best_score:
                best_score = score
                best_candidate = aggregator

                current_opt = copy.deepcopy(self.opt)
                for k in aggregator.opt:
                    current_opt[k] = best_candidate.opt[k]

        print("[RESULT] Best AUROC-Score: {}".format(best_score))
        for k in best_candidate.opt:
            print("[RESULT] ", k, self.opt[k], " --> ", best_candidate.opt[k])
            self.opt[k] = best_candidate.opt[k]
        return best_score


class OutlierDetectorSearch:
    def __init__(self, opt, class_mapping):
        self.opt = opt
        self.class_mapping = class_mapping

        self.feature_opt = ["feature", "sampling_method", "sampling_step", "sampling_window", "image_size"]

        self.image_classifier = None

    def new(self):

        for k in self.feature_opt:
            if k not in self.opt:
                continue
            if type(self.opt[k]) is not list:
                self.opt[k] = [self.opt[k]]

        feature_opt_list = list(ParameterGrid({k: self.opt[k] for k in self.feature_opt if k in self.opt}))

        for opt in feature_opt_list:
            for k in self.opt:
                if k not in opt:
                    opt[k] = self.opt[k]

        self.image_classifier = [OutlierDetector(opt, self.class_mapping) for opt in feature_opt_list]

    def fit(self, model_folder, data_path_known, data_path_test, tag_type, report_path=None):
        self.new()

        best_score = 0
        best_candidate = None

        check_n_make_dir(model_folder)

        for i, cls in enumerate(self.image_classifier):
            score = cls.fit(
                os.path.join(model_folder, "version_{}".format(i)),
                data_path_known,
                data_path_test,
                tag_type,
                report_path=report_path
            )

            if score > best_score:
                best_candidate = i
                best_score = score

        print("[RESULT] Best Model: {} ({})".format(best_candidate, best_score))
        for f in os.listdir(os.path.join(model_folder, "version_{}".format(best_candidate))):
            shutil.copy(
                os.path.join(model_folder, "version_{}".format(best_candidate), f),
                os.path.join(model_folder, f))
        return best_score


