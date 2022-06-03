import argparse
import os
import copy
from classic_image_classification.machine_learning.optimizing_image_classifier import OptimizingImageClassifier

from test_image_classifier import test
import numpy as np
from classic_image_classification.utils.utils import load_dict


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = None
        self.mf = model_folder

        self.opt = {
            "data_split_mode": "random",
            "aggregator": "bag_of_words",
            "complexity": [8, 16, 32, 64, 128, 256, 512],
            "type": ["rf", "xgboost"],
            "feature": "hsv-hog",
            "sampling_method": "dense",
            "sampling_step": 16,
            "sampling_window": 16,
            "image_size": {
                "width": 256,
                "height": 256,
            },
        }


def start_training(args_, cfg):
    df = args_.dataset_folder
    mf = cfg.mf
    dtype = args_.dataset_type

    split = 0.25

    image_cls = OptimizingImageClassifier(cfg.opt, cfg.class_mapping)

    image_cls.fit(df, dtype, load_all=False)
    image_cls.save(mf)
    if args_.test_folder is not None:
        test(mf, args_.test_folder, dt=args_.dataset_type)


def main(args_):
    cfg = Config(args_.model_folder)
    # cfg.class_mapping = load_dict(args_.class_mapping)
    cfg.class_mapping = {
        "cobblestone": 0,
        "bg": 1,
    }
    print(cfg.class_mapping)
    start_training(args_, cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with dataset",
    )
    parser.add_argument(
        "--test_folder",
        "-tf",
        default=None,
        help="Path to directory with test dataset",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    parser.add_argument(
        "--model_folder",
        "-model",
        help="Path to model",
    )
    parser.add_argument(
        "--class_mapping",
        "-clmp",
        help="Path to class mapping JSON",
    )
    parser.add_argument(
        "--use_cache",
        "-cache",
        type=bool,
        default=False,
        help="Save the Calculated Features in _cache folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
