import os
import joblib
import numpy as np
from time import time
import logging

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from classic_image_classification.utils.utils import check_n_make_dir, save_dict, load_dict


def init_ensembles(clf_type):
    if clf_type == "rf_100":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)
    elif clf_type == "rf_200":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
    elif clf_type == "rf_500":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    elif clf_type == "rf_1000":
        return RandomForestClassifier(n_estimators=1000, class_weight="balanced", n_jobs=-1)
    elif clf_type == "rf_2000":
        return RandomForestClassifier(n_estimators=2000, class_weight="balanced", n_jobs=-1)
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)


def init_knn(clf_type):
    if clf_type == "knn_3":
        return KNeighborsClassifier(n_neighbors=3)
    elif clf_type == "knn_5":
        return KNeighborsClassifier(n_neighbors=5)
    elif clf_type == "knn_7":
        return KNeighborsClassifier(n_neighbors=7)
    elif clf_type == "knn_9":
        return KNeighborsClassifier(n_neighbors=9)
    return KNeighborsClassifier()


def init_mlp(clf_type):
    max_iter = 10000
    if clf_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=max_iter)
    elif clf_type == "mlp_x":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=max_iter)
    elif clf_type == "mlp_xx":
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=max_iter)
    elif clf_type == "mlp_xxx":
        return MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64), max_iter=max_iter)
    return MLPClassifier(max_iter=max_iter)


def init_other(clf_type):
    if clf_type == "gp":
        return GaussianProcessClassifier((1.0 * kernels.RBF(1.0)), n_jobs=-1)
    elif clf_type == "lr":
        return LogisticRegressionCV(class_weight='balanced', max_iter=10000)
    elif clf_type == "svm":
        return SVC(kernel='rbf', class_weight='balanced', gamma="scale")
    elif clf_type == "nc":
        return NearestCentroid()
    raise Exception("Unknown Classifier Option: {}.".format(clf_type))


def init_classifier(opt):
    if "rf" in opt["type"]:
        return init_ensembles(opt["type"])
    elif "mlp" in opt["type"]:
        return init_mlp(opt["type"])
    elif "knn" in opt["type"]:
        return init_knn(opt["type"])
    else:
        return init_other(opt["type"])


class Classifier:
    def __init__(self, opt=None, class_mapping=None):
        self.opt = opt
        self.class_mapping = class_mapping
        self.class_mapping_inv = None

        self.classifier = None

    def __str__(self):
        return "Classifier: {}".format(self.opt["type"])

    def fit(self, x_train, y_train):
        self.new()
        logging.info("Fitting the {} to the training set".format(self.opt["type"]))
        t0 = time()
        self.classifier.fit(x_train, y_train)
        logging.info("done in %0.3fs" % (time() - t0))

    def predict(self, x, get_confidence=False):
        if get_confidence:
            if self.opt["type"] in ["nc", "svm"]:
                return self.classifier.predict(x), 1.0
            prob = self.classifier.predict_proba(x)
            return [np.argmax(prob)], np.max(prob)
        return self.classifier.predict(x)

    def evaluate(self, x_test, y_test, save_path=None, print_results=True):
        logging.info("Predicting on the test set")
        t0 = time()
        y_pred = self.predict(x_test)
        logging.info("done in %0.3fs" % (time() - t0))

        s = ""
        s += str(classification_report(y_test, y_pred, zero_division=0))
        s += "\nConfusion Matrix:\n"
        s += str(confusion_matrix(y_test, y_pred))
        if print_results:
            print(s)
        if save_path is not None:
            d = os.path.dirname(save_path)
            if not os.path.isdir(d):
                os.mkdir(d)
            with open(save_path, "w") as f:
                f.write(s)

        return f1_score(y_true=y_test, y_pred=y_pred, average="macro")

    def new(self):
        self.classifier = init_classifier(self.opt)

    def load(self, model_path):
        path_to_class_mapping = os.path.join(model_path, "class_mapping.json")
        path_to_pipeline_opt = os.path.join(model_path, "pipeline_opt.json")
        path_to_classifier = os.path.join(model_path, "classifier.pkl")
        self.class_mapping = load_dict(path_to_class_mapping)
        self.opt = load_dict(path_to_pipeline_opt)
        self.classifier = joblib.load(path_to_classifier)
        if self.class_mapping is not None:
            self.class_mapping_inv = {v: k for k, v in self.class_mapping.items()}
        logging.info("Classifier was loaded!")

    def save(self, model_path):
        path_to_class_mapping = os.path.join(model_path, "class_mapping.json")
        path_to_pipeline_opt = os.path.join(model_path, "pipeline_opt.json")
        path_to_classifier = os.path.join(model_path, "classifier.pkl")
        check_n_make_dir(model_path, clean=False)
        save_dict(self.class_mapping, path_to_class_mapping)
        save_dict(self.opt, path_to_pipeline_opt)
        if self.classifier is not None:
            joblib.dump(self.classifier, path_to_classifier)
