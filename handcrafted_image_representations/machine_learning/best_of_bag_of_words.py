import os
import shutil
from sklearn.model_selection import ParameterGrid
from handcrafted_image_representations import machine_learning as ml
from handcrafted_image_representations.utils.utils import check_n_make_dir
from handcrafted_image_representations.data_structure.data_set import DataSet


class BestOfBagOfWords:
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

        self.image_classifier = [ml.OptimizingImageClassifier(opt, self.class_mapping) for opt in feature_opt_list]

    def fit(self, model_folder, tags, report_path=None):
        self.new()

        best_f1_score = 0
        best_candidate = None

        check_n_make_dir(model_folder)

        for i, cls in enumerate(self.image_classifier):
            score = cls.fit(
                os.path.join(model_folder, "version_{}".format(i)),
                tags,
                report_path=report_path
            )

            if score > best_f1_score:
                best_candidate = i
                best_f1_score = score

        print("[RESULT] Best Model: {} ({})".format(best_candidate, best_f1_score))
        for f in os.listdir(os.path.join(model_folder, "version_{}".format(best_candidate))):
            shutil.copy(
                os.path.join(model_folder, "version_{}".format(best_candidate), f),
                os.path.join(model_folder, f))
        return best_f1_score

    def fit_folder(self, model_folder, data_path, tag_type="cls", load_all=False, report_path=None):
        ds = DataSet(data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        return self.fit(model_folder, tags, report_path)
