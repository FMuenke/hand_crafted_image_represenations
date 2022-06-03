import os
import numpy as np
from classic_image_classification.machine_learning.classic_image_classifier import ClassicImageClassifier


class EnsembleImageClassifier:
    def __init__(self, path_to_ensemble_members):
        self.path_to_ensemble_members = path_to_ensemble_members
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

    def load(self):
        print("Loading Ensemble Members...")
        self.load_model(self.path_to_ensemble_members)
        for f in os.listdir(self.path_to_ensemble_members):
            model_path = os.path.join(self.path_to_ensemble_members, f)
            if os.path.isdir(model_path):
                self.load_model(model_path)

        print("{} Members were Loaded:".format(len(self.members)))
        for k in self.members:
            print("- {}".format(k))

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




