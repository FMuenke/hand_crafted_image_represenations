import argparse
import os
import numpy as np
from datastructure.data_set import DataSet
from machine_learning.classic_image_classifier import ClassicImageClassifier
from datastructure.tag_gate import TagGate

from datastructure.data_saver import DataSaver

from test_image_classifier import test

import utils.parameter_grid as pg


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.90

        self.class_mapping = {
            # "gas": 0,
            # "leaf_dirt": 0,
            # "outburst_0": 1,
            "surface_patch_0": 0,
            "surface_patch_1": 1,
        }

        self.opt = {
            "data_split_mode": "random",
            "classifier_opt": {
                "aggregator": "fisher_vector",
                "complexity": 500,
                "type": "random_forrest",
                # "n_estimators": 600,
                "param_grid": pg.random_forrest_grid(),
            },
            "feature": ["gray-kaze"],
            "sampling_method": "kaze",
            "sampling_step": 0,
            "sampling_window": 0,
            "image_size": {
                # "roi": [0.35, 0.5, 0.5, 0.99],
                "width": None,
                "height": None,
                "padding": False,
            },
        }
        self.mf = "{}_{}_{}_{}_{}_{}_{}".format(model_folder,
                                                self.opt["feature"],
                                                self.opt["sampling_method"],
                                                self.opt["sampling_step"],
                                                self.opt["sampling_window"],
                                                self.opt["classifier_opt"]["aggregator"],
                                                self.opt["classifier_opt"]["complexity"])

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
    if args_.test_folder is not None:
        split = 0.1

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

    ml_pipeline.build_aggregator(x)
    x = ml_pipeline.aggregate(x)

    if args_.use_cache:
        cache = DataSaver(os.path.join(df, "_cache"))
        cache.add("x", np.concatenate(x, axis=0))
        cache.add("y", y)

    f_1_score = ml_pipeline.fit(x, y, percentage=split)
    ml_pipeline.save()

    if args_.test_folder is not None:
        test(mf, args_.test_folder, dt=args_.dataset_type)

    return f_1_score


def main(args_):
    cfg = Config(args_.model_folder)
    f_1 = start_training(args_, cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
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
        "-m",
        default="./test/",
        help="Path to model",
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
