import argparse
import os
import copy
from datastructure.data_set import DataSet
from machine_learning.classic_image_classifier import ClassicImageClassifier
from datastructure.tag_gate import TagGate

from datastructure.data_saver import DataSaver

from test_image_classifier import test

import utils.parameter_grid as pg
from utils.utils import load_dict


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = None

        self.opt = {
            "data_split_mode": "random",
            "classifier_opt": {
                "aggregator": "bag_of_words",
                "complexity": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                "type": "svm",
                # "n_estimators": 5000,
                # "param_grid": pg.support_vector_machine_grid(),
            },
            "feature": ["hsv-hog+32+L2"],
            "sampling_method": "dense",
            "sampling_step": 16,
            "sampling_window": 16,
            "image_size": {
                # "roi": [0.35, 0.5, 0.5, 0.99],
                "width": None,
                "height": None,
                "padding": False,
            },
        }
        self.mf = "{}_{}_{}_{}_{}_{}".format(model_folder,
                                             self.opt["feature"],
                                             self.opt["sampling_method"],
                                             self.opt["sampling_step"],
                                             self.opt["sampling_window"],
                                             self.opt["classifier_opt"]["aggregator"])

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
                    class_mapping=cfg.class_mapping)

    d_set.load_data(tag_type=dtype)
    assert len(d_set.tags) != 0, "No Labels were found! Abort..."

    tag_set = d_set.get_tags()

    tg = TagGate({"height": 0, "width": 0}, cfg.down_sample)
    tag_set = tg.apply(tag_set)

    ml_pipeline = ClassicImageClassifier(model_path=mf,
                                         pipeline_opt=cfg.opt,
                                         class_mapping=cfg.class_mapping)

    ml_pipeline.new()
    x, y = ml_pipeline.extract(tag_set)

    cache = DataSaver(os.path.join(df, "_cache"))
    cache.add("x", x)
    cache.add("y", y)

    best_f1_score = 0
    best_comp = None
    best_candidate = None

    if "complexity" in cfg.opt["classifier_opt"]:
        agg_complexities = cfg.opt["classifier_opt"]["complexity"]
    else:
        agg_complexities = 0
    if not type(agg_complexities) is list:
        agg_complexities = [agg_complexities]

    for complexity in agg_complexities:
        candidate_opt = copy.deepcopy(cfg.opt)
        candidate_opt["classifier_opt"]["complexity"] = complexity

        candidate = ClassicImageClassifier(model_path=mf, pipeline_opt=candidate_opt, class_mapping=cfg.class_mapping)
        candidate.new()

        x = cache.get_special("x")
        y = cache.get("y")

        candidate.build_aggregator(x)
        x = candidate.aggregate(x)

        f_1_score = candidate.fit(x, y, percentage=split)
        if f_1_score > best_f1_score:
            best_f1_score = f_1_score
            best_candidate = candidate
            best_comp = complexity

    print("Best Complexity for {} was: {}".format(cfg.opt["classifier_opt"]["aggregator"], best_comp))
    best_candidate.save()
    cache.clear_storage()

    if args_.test_folder is not None:
        test(mf, args_.test_folder, dt=args_.dataset_type)

    return best_f1_score


def main(args_):
    cfg = Config(args_.model_folder)
    cfg.class_mapping = load_dict(args_.class_mapping)
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
