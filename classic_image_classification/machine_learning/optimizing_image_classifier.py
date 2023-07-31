import copy
import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification import machine_learning as ml

from classic_image_classification.utils.data_split import split_tags


class OptimizingImageClassifier:
    def __init__(self, opt, class_mapping):
        self.opt = opt
        self.class_mapping = class_mapping

        self.aggregator_opt = ["aggregator", "complexity"]
        self.classifier_opt = ["type", "n_estimators"]

        self.feature_extractor = None
        self.aggregator_list = None
        self.classifier_list = None

        self.final_classifier = None

    def new(self):
        if "sampling_step" not in self.opt:
            self.opt["sampling_step"] = 20
        if "sampling_window" not in self.opt:
            self.opt["sampling_window"] = 20

        self.feature_extractor = ml.FeatureExtractor(
            features_to_use=self.opt["feature"],
            image_height=self.opt["image_size"]["height"],
            image_width=self.opt["image_size"]["width"],
            sampling_method=self.opt["sampling_method"],
            sampling_steps=self.opt["sampling_step"],
            sampling_window=self.opt["sampling_window"]
        )

        for k in self.aggregator_opt + self.classifier_opt:
            if k not in self.opt:
                continue
            if type(self.opt[k]) is not list:
                self.opt[k] = [self.opt[k]]

        aggregator_opt_list = list(ParameterGrid({k: self.opt[k] for k in self.aggregator_opt if k in self.opt}))
        classifier_opt_list = list(ParameterGrid({k: self.opt[k] for k in self.classifier_opt if k in self.opt}))

        self.aggregator_list = [ml.Aggregator(opt) for opt in aggregator_opt_list]
        self.classifier_list = [ml.Classifier(opt, self.class_mapping) for opt in classifier_opt_list]

    def fit(self, model_folder, data_path, tag_type, load_all=False, report_path=None):
        self.new()
        best_f1_score = 0
        best_candidate = None

        ds = DataSet(data_path, tag_type, self.class_mapping)
        ds.load_data()
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)

        train_tags, test_tags = split_tags(tags, mode=self.opt["data_split_mode"])

        x_train, y_train = self.feature_extractor.extract_trainings_data(train_tags)
        x_test, y_test = self.feature_extractor.extract_trainings_data(test_tags)

        for aggregator in self.aggregator_list:
            x_transformed_train = aggregator.fit_transform(x_train)
            x_transformed_test = aggregator.transform(x_test)

            x_transformed_train = np.concatenate(x_transformed_train, axis=0)
            x_transformed_test = np.concatenate(x_transformed_test, axis=0)

            for cls in self.classifier_list:
                cls.fit(x_transformed_train, y_train)
                f_1_score = cls.evaluate(x_transformed_test, y_test, print_results=False)
                if f_1_score > best_f1_score:
                    best_f1_score = f_1_score
                    best_candidate = [aggregator, cls]

                    current_opt = copy.deepcopy(self.opt)
                    for k in aggregator.opt:
                        current_opt[k] = best_candidate[0].opt[k]
                    for k in cls.opt:
                        current_opt[k] = best_candidate[1].opt[k]

                    self.final_classifier = ml.ClassicImageClassifier(opt=current_opt, class_mapping=self.class_mapping)
                    self.final_classifier.feature_extractor = self.feature_extractor
                    self.final_classifier.aggregator = aggregator
                    self.final_classifier.classifier = cls
                    self.final_classifier.save(model_folder)

        print("[RESULT] Best F1-Score: {}".format(best_f1_score))
        for k in best_candidate[0].opt:
            print("[RESULT] ", k, self.opt[k], " --> ", best_candidate[0].opt[k])
            self.opt[k] = best_candidate[0].opt[k]
        for k in best_candidate[1].opt:
            print("[RESULT] ", k, self.opt[k], " --> ", best_candidate[1].opt[k])
            self.opt[k] = best_candidate[1].opt[k]
        return best_f1_score

