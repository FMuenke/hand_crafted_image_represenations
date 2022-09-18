import argparse
import os
import numpy as np
from tqdm import tqdm
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification.machine_learning import ClassicImageClassifier
from classic_image_classification.machine_learning import ImageEmbedding
from classic_image_classification.utils.utils import check_n_make_dir

from classic_image_classification.machine_learning.outlier_detector import OutlierDetectorSearch

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, classification_report
from classic_image_classification.utils.outlier_removal import get_best_threshold


def make_displot(data_frame, key, model_folder):
    sns.displot(data=data_frame, x=key, hue="status", kind="kde")
    plt.savefig(os.path.join(model_folder, "{}.png".format(key)))
    plt.close()


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = {
            "manhole": 0,
            "stormdrain": 1,
        }
        self.mf = model_folder

        self.opt = {
            "data_split_mode": "random",
            "aggregator": "bag_of_words",
            "complexity": [8, 16, 32, 64, 128, 256, 512],
            "type": ["rf", "xgboost"],
            "feature": ["hsv-hog"],
            "sampling_method": "dense",
            "sampling_step": [16, 32],
            "sampling_window": [32],
            "image_size": [
                {
                    "width": 128,
                    "height": 128,
                }
            ]
        }


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    dt = args_.dataset_type
    tf = args_.test_data_folder

    cfg = Config(mf)

    od = OutlierDetectorSearch(cfg.opt, cfg.class_mapping)
    od.fit(mf, data_path_known=df, data_path_test=tf, tag_type=dt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with known data to fit the initial model",
    )
    parser.add_argument(
        "--model_folder",
        "-m",
        help="Path to save the model to",
    )
    parser.add_argument(
        "--test_data_folder",
        "-tf",
        default=None,
        help="Path to directory with known and unknown data to test on",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
