import logging
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from handcrafted_image_representations.utils.utils import save_dict, load_dict, check_n_make_dir

from handcrafted_image_representations.machine_learning.feature_extractor import FeatureExtractor
from handcrafted_image_representations.machine_learning.aggregator import Aggregator

from handcrafted_image_representations.data_structure.data_set import DataSet


class ImageEmbedding:
    def __init__(self, 
                 aggregator="bag_of_words", 
                 complexity=1024, 
                 feature="hsv-sift",
                 sampling_method="dense",
                 sampling_window=16,
                 sampling_step=16,
                 image_size_width=128,
                 image_size_height=128,
                 ):
        
        self.opt = {
            "aggregator": aggregator,
            "complexity": complexity,
            "feature": feature,
            "sampling_method": sampling_method,
            "sampling_window": sampling_window,
            "sampling_step": sampling_step,
            "image_size": {
                "width": image_size_width, 
                "height": image_size_height
                }
        }
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

    def fit_folder(self, data_path, tag_type, classes_to_consider="all"):
        ds = DataSet(data_path, tag_type=tag_type)
        tags = ds.get_tags(classes_to_consider=classes_to_consider)

        self.new()
        x, _ = self.feature_extractor.extract_trainings_data(tags)
        self.feature_aggregator.fit(x)

    def fit(self, tags):
        self.new()
        x, _ = self.feature_extractor.extract_trainings_data(tags)
        self.feature_aggregator.fit(x)

    def fit_transform(self, tags, return_y=False):
        self.new()
        x, y = self.feature_extractor.extract_trainings_data(tags)
        x_trans = self.feature_aggregator.fit_transform(x)
        if return_y:
            return x_trans, y
        return x_trans
    
    def fit_and_sort_folder(self, data_path, export_path, tag_type, n_clusters, classes_to_consider="all"):
        ds = DataSet(data_path, tag_type=tag_type)
        tags = ds.get_tags(classes_to_consider=classes_to_consider)
        sorted_tags = self.fit_and_sort(tags, n_clusters)
        os.makedirs(export_path, exist_ok=True)
        for cl in sorted_tags:
            cluster_path = os.path.join(export_path, str(cl))
            os.makedirs(cluster_path, exist_ok=True)
            for tag in sorted_tags[cl]:
                tag.export_box(cluster_path)
    
    def fit_and_sort(self, tags, n_clusters):
        self.new()
        x, _ = self.feature_extractor.extract_trainings_data(tags)
        x_trans = self.feature_aggregator.fit_transform(x)
        x_trans = np.concatenate(x_trans, axis=0)
        kmean = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto")
        y = kmean.fit_predict(x_trans)
        sorted_tags = {}
        for cl, tag in zip(y, tags):
            if cl not in sorted_tags:
                sorted_tags[cl] = []
            sorted_tags[cl].append(tag)
        return sorted_tags

    def transform(self, image):
        x = self.feature_extractor.extract_x(image)
        x = self.feature_aggregator.transform(x)
        return x[0]

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
