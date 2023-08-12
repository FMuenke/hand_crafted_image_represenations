import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA

from classic_image_classification.utils.utils import save_dict, load_dict, check_n_make_dir

from classic_image_classification.machine_learning.feature_extractor import FeatureExtractor
from classic_image_classification.machine_learning.aggregator import Aggregator

from classic_image_classification.data_structure.data_set import DataSet


class ImageEmbedding:
    def __init__(self, opt):
        self.opt = opt
        self.feature_extractor = None
        self.feature_aggregator = None

        self.data_set_repr = None
        self.data_set_tags = None

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

    def is_fitted(self):
        if self.feature_aggregator is None or self.feature_extractor is None:
            return False
        return True

    def register_data_set(self, data_path, tag_type, classes_to_consider="all"):
        ds = DataSet(data_path, tag_type=tag_type)
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        self.data_set_tags = tags

        if self.is_fitted():
            x, y = self.feature_extractor.extract_trainings_data(tags)
            self.data_set_repr = self.feature_aggregator.transform(x)
        else:
            self.new()
            x, y = self.feature_extractor.extract_trainings_data(tags)
            self.data_set_repr = self.feature_aggregator.fit_transform(x)
        self.data_set_repr = np.concatenate(self.data_set_repr, axis=0)

    def show(self):
        x = np.concatenate(self.data_set_repr, axis=0)

        projection = PCA(n_components=4)
        x_proj = projection.fit_transform(x)

        df = {"name": []}
        for t_id in self.data_set_tags:
            df["name"].append(self.data_set_tags[t_id].tag_class[0])

        list_of_variables = []
        for i in range(x_proj.shape[1]):
            list_of_variables.append("x{}".format(i + 1))
            df["x{}".format(i + 1)] = x_proj[:, i]

        df = pd.DataFrame(df)
        sns.pairplot(data=df, vars=list_of_variables, hue="name", kind="kde")
        plt.show()

    def query(self, image, n=3):
        assert self.data_set_tags is not None, "No Tags are registered."
        assert self.data_set_repr is not None, "No Representations are computed."

        x_trans = self.transform(image)
        distances = np.sqrt(np.sum(np.square(self.data_set_repr - x_trans), axis=1))
        sorted_indices = np.argsort(distances)
        selected_tags_match = [self.data_set_tags[i] for i in sorted_indices[:n]]
        selected_tags_mismatch = [self.data_set_tags[i] for i in sorted_indices[-n:]]
        return selected_tags_match, selected_tags_mismatch

    def fit(self, data_path, tag_type, classes_to_consider="all"):
        ds = DataSet(data_path, tag_type=tag_type)
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        self.new()
        x, _ = self.feature_extractor.extract_trainings_data(tags)
        self.feature_aggregator.fit(x)

    def transform(self, image):
        x = self.feature_extractor.extract_x(image)
        x = self.feature_aggregator.transform(x)
        return x[0]

    def fit_transform(self, data_path, tag_type, classes_to_consider="all", return_y=False):
        ds = DataSet(data_path, tag_type=tag_type)
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        self.new()
        x, y = self.feature_extractor.extract_trainings_data(tags)
        x_trans = self.feature_aggregator.fit_transform(x)
        if return_y:
            return x_trans, y
        return x_trans

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
