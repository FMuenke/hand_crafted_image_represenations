import argparse
import os
import copy
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification.machine_learning.classic_image_classifier import ClassicImageClassifier

from classic_image_classification.data_structure.data_saver import DataSaver

from test_image_classifier import test
import numpy as np
from classic_image_classification.utils.utils import load_dict


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = None

        self.opt = {
            "data_split_mode": "random",
            "aggregator": "bag_of_words",
            "complexity": [8, 16, 32, 64, 128, 256, 512],
            "type": "svm",
            "feature": "hsv-hog",
            "sampling_method": "dense",
            "sampling_step": 16,
            "sampling_window": 16,
            "image_size": {
                "width": 64,
                "height": 64,
                "padding": False,
            },
        }
        self.mf = "{}_{}_{}_{}_{}_{}".format(model_folder,
                                             self.opt["feature"],
                                             self.opt["sampling_method"],
                                             self.opt["sampling_step"],
                                             self.opt["sampling_window"],
                                             self.opt["aggregator"])

        self.mf = self.mf.replace("[", "")
        self.mf = self.mf.replace("]", "")
        self.mf = self.mf.replace("'", "")
        self.mf = self.mf.replace(",", "_")
        self.mf = self.mf.replace(" ", "")


def start_training(args_, cfg):
    df = args_.dataset_folder
    mf = cfg.mf
    dtype = args_.dataset_type

    split = 0.25

    d_set = DataSet(data_set_dir=df,
                    tag_type=dtype,
                    class_mapping=cfg.class_mapping)

    d_set.load_data()
    assert len(d_set.tags) != 0, "No Labels were found! Abort..."

    tag_set = d_set.get_tags()

    ml_pipeline = ClassicImageClassifier(opt=cfg.opt,
                                         class_mapping=cfg.class_mapping)

    ml_pipeline.new()
    x, y = ml_pipeline.extract(tag_set)

    cache = DataSaver(os.path.join(df, "_cache"))
    cache.add("x", x)
    cache.add("y", y)

    best_f1_score = 0
    best_comp = None
    best_candidate = None

    if "complexity" in cfg.opt:
        agg_complexities = cfg.opt["complexity"]
    else:
        agg_complexities = 0
    if not type(agg_complexities) is list:
        agg_complexities = [agg_complexities]

    for complexity in agg_complexities:
        candidate_opt = copy.deepcopy(cfg.opt)
        candidate_opt["complexity"] = complexity

        candidate = ClassicImageClassifier(opt=candidate_opt, class_mapping=cfg.class_mapping)
        candidate.new()

        x = cache.get_special("x")
        y = cache.get("y")

        candidate.build_aggregator(x)
        print(np.array(x).shape)
        x = candidate.aggregate(x)

        f_1_score = candidate.fit(x, y)
        if f_1_score > best_f1_score:
            best_f1_score = f_1_score
            best_candidate = candidate
            best_comp = complexity

    print("Best Complexity for {} was: {}".format(cfg.opt["classifier_opt"]["aggregator"], best_comp))
    best_candidate.save(mf)
    cache.clear_storage()

    if args_.test_folder is not None:
        test(mf, args_.test_folder, dt=args_.dataset_type)

    return best_f1_score


def main(args_):
    cfg = Config(args_.model_folder)
    # cfg.class_mapping = load_dict(args_.class_mapping)
    cfg.class_mapping = {
        "speed_20": 0,
        "speed_30": 1,
        "speed_50": 2,
        "speed_60": 3,
        "speed_70": 4,
        "speed_80": 5,
        "speed_100": 6,
        "speed_120": 7,
        "speed_40": 8,
        "bg": 9,
    }
    print(cfg.class_mapping)
    f_1 = start_training(args_, cfg)


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
