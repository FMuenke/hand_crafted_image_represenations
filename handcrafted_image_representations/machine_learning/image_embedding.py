import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from handcrafted_image_representations.utils.utils import save_dict, load_dict, check_n_make_dir

from handcrafted_image_representations.machine_learning.feature_extractor import FeatureExtractor
from handcrafted_image_representations.machine_learning.aggregator import Aggregator

from handcrafted_image_representations.data_structure.data_set import DataSet


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

    def show(self, path_to_store=None):
        assert self.data_set_tags is not None, "No Tags are registered."
        assert self.data_set_repr is not None, "No Representations are computed."

        projection = PCA(n_components=2)
        x_proj = projection.fit_transform(self.data_set_repr)

        df = {"name": []}
        for tag in self.data_set_tags:
            df["name"].append(tag.tag_class[0])

        df["x1"] = x_proj[:, 0]
        df["x2"] = x_proj[:, 1]

        df = pd.DataFrame(df)
        plt.title("Distribution")
        sns.scatterplot(data=df, x="x1", y="x2", hue="name")
        if path_to_store is None:
            plt.show()
        else:
            plt.savefig(path_to_store)
            plt.close()

    def query(self, image, n=3):
        assert self.data_set_tags is not None, "No Tags are registered."
        assert self.data_set_repr is not None, "No Representations are computed."

        x_trans = self.transform(image)
        distances = np.sqrt(np.sum(np.square(self.data_set_repr - x_trans), axis=1))
        sorted_indices = np.argsort(distances)
        selected_tags_match = [self.data_set_tags[i] for i in sorted_indices[:n]]
        selected_tags_mismatch = [self.data_set_tags[i] for i in sorted_indices[-n:]]
        return selected_tags_match, selected_tags_mismatch

    def sample(self, n):
        assert self.data_set_tags is not None, "No Tags are registered."
        assert self.data_set_repr is not None, "No Representations are computed."

        clustering = MiniBatchKMeans(n_clusters=n, n_init="auto")
        clustering.fit(self.data_set_repr)
        cluster_assignments = clustering.labels_

        cluster_indices = {}
        for idx, label in enumerate(cluster_assignments):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)

        # Randomly sample one instance from each cluster
        sampled_indices = []
        for cluster_label, indices in cluster_indices.items():
            sampled_indices.append(np.random.choice(indices))

        selected_tags = [self.data_set_tags[i] for i in sampled_indices]
        return selected_tags

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
