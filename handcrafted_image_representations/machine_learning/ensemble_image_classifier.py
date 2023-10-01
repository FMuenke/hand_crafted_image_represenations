import os
import numpy as np
import logging
from tqdm import tqdm
from handcrafted_image_representations.machine_learning.classic_image_classifier import ClassicImageClassifier
from handcrafted_image_representations.data_structure.data_set import DataSet
from handcrafted_image_representations.utils.utils import check_n_make_dir
from handcrafted_image_representations.utils.statistic_utils import init_result_dict, show_results, save_results


class EnsembleImageClassifier:
    def __init__(self):
        self.class_mapping = None
        self.class_mapping_inv = None

        self.members = {}

        self.multi_class = False

    def load_model(self, model_path):
        if os.path.isfile(os.path.join(model_path, "classifier.pkl")):
            c = ClassicImageClassifier(model_path)
            c.load(model_path)
            k = "{}_classic".format(len(self.members))
            self.members[k] = c

    def load(self, path_to_ensemble_members):
        logging.info("Loading Ensemble Members...")
        self.load_model(path_to_ensemble_members)
        for f in os.listdir(path_to_ensemble_members):
            model_path = os.path.join(path_to_ensemble_members, f)
            if os.path.isdir(model_path):
                self.load_model(model_path)

        logging.info("{} Members were Loaded:".format(len(self.members)))
        for k in self.members:
            logging.info("- {}".format(k))

        self.load_class_mapping()

    def load_class_mapping(self):
        clmp = {}
        for k in self.members:
            m = self.members[k]
            for c in m.class_mapping:
                if c not in clmp:
                    clmp[c] = m.class_mapping[c]
                else:
                    assert clmp[c] == m.class_mapping[c], "Class Mappings have to match each other!"
        self.class_mapping = clmp
        self.class_mapping_inv = {v: k for k, v in self.class_mapping.items()}

    def predict_image(self, img, get_confidence=False):
        votes = []
        confs = []
        for k in self.members:
            y_pred, conf = self.members[k].predict_image(img, get_confidence=True)
            votes.append(y_pred)
            confs.append(conf)
        votes = np.array(votes)
        unique, counts = np.unique(votes, return_counts=True)
        if self.multi_class:
            return unique
        voted = np.argmax(counts)

        if get_confidence:
            return votes[voted], np.mean(confs)
        else:
            return votes[voted]

    def evaluate(self, data_path, tag_type, load_all=False, report_path=None):
        ds = DataSet(data_set_dir=data_path, class_mapping=self.class_mapping, tag_type=tag_type)
        ds.load_data()
        if load_all:
            tags = ds.get_tags()
        else:
            tags = ds.get_tags(self.class_mapping)
        assert len(tags) != 0, "No Labels were found! Abort..."

        if report_path is not None:
            check_n_make_dir(report_path, clean=True)

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
