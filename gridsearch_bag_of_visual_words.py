import argparse
from time import sleep
from sklearn.model_selection import ParameterGrid

from train_image_classifier import start_training

import utils.parameter_grid as pg


class ConfigBuilder:
    def __init__(self, model_folder):
        self.mf = model_folder
        self.opt = {
            "data_split_mode": ["random"],
            "classifier_opt": [
                {
                    "aggregator": "bag_of_words",
                    "complexity": 20,
                    "type": "b_bagging",
                    "param_grid": pg.bagging_grid(),
                },
                {
                    "aggregator": "bag_of_words",
                    "complexity": 250,
                    "type": "b_bagging",
                    "param_grid": pg.bagging_grid(),
                },
            ],
            "feature": [["gray-hog"]],
            "sampling_method": ["dense"],
            "sampling_step": [2],
            "sampling_window": [5],
            "image_size": [
                {
                    "name": "400",
                    "roi": [0.35, 0.6, 0.5, 0.99],
                    "width": 400,
                    "height": 400,
                    "padding": True,
                },
            ],
        }

        self.num_opt = 0

    def list_of_cfg(self):
        list_of_configs = []
        for opt in list(ParameterGrid(self.opt)):
            print(opt)
            cfg = Config(opt, self.mf)
            list_of_configs.append(cfg)

        self.num_opt = len(list_of_configs)
        return list_of_configs


class Config:
    def __init__(self, opt, model_folder):
        self.class_mapping = {
            "cobblestone": 1,
            "bg": 2
        }

        self.opt = opt
        self.mf = "{}_feat-{}_{}_{}_{}_cls-{}_{}_{}_img-{}".format(
            model_folder,
            self.opt["feature"],
            self.opt["sampling_method"],
            self.opt["sampling_step"],
            self.opt["sampling_window"],
            self.opt["classifier_opt"]["type"],
            self.opt["classifier_opt"]["aggregator"],
            self.opt["classifier_opt"]["complexity"],
            self.opt["image_size"]["name"]
        )


def main(args_):
    cfg_builder = ConfigBuilder(args_.model)
    for idx, cfg in enumerate(cfg_builder.list_of_cfg()):
        print("----------------------------------")
        print("Testing Config:")
        print("Run {} / {}".format(idx+1, cfg_builder.num_opt))
        print(cfg.opt)
        try:
            score = start_training(args_, cfg)
        except Exception as e:
            print(" ")
            print("Does not Work")
            print(e)
            score = 0
            sleep(5)
        print("The config reached a f1 score of {}".format(score))
        print("----------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="./test/",
        help="Path to model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
