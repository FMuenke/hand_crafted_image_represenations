import argparse
import os
import copy
from datastructure.data_set import DataSet
from machine_learning.classic_image_classifier import ClassicImageClassifier
from datastructure.tag_gate import TagGate

from datastructure.data_saver import DataSaver

from test_image_classifier import test

import utils.parameter_grid as pg


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = {
            "Kyberge_blanket1": 0,
            "Kyberge_canvas1": 1,
            "Kyberge_seat2": 2,
            "UIUC07_water": 3,
            "UIUC02_bark2": 4,
            "KTH_brown_bread": 5,
            "UIUC17_glass2": 6,
            "Kyberge_scarf1": 7,
            "KTH_corduroy": 8,
            "UIUC16_glass1": 9,
            "Kyberge_stoneslab1": 10,
            "Kyberge_rice2": 11,
            "UIUC06_wood3": 12,
            "KTH_aluminium_foil": 13,
            "Kyberge_ceiling1": 14,
            "Kyberge_sesameseeds1": 15,
            "Kyberge_floor2": 16,
            "Kyberge_lentils1": 17,
            "KTH_linen": 18,
            "UIUC08_granite": 19,
            "Kyberge_screen1": 20,
            "UIUC24_corduroy": 21,
            "Kyberge_oatmeal1": 22,
            "Kyberge_stone1": 23,
            "UIUC03_bark3": 24,
            "Kyberge_pearlsugar1": 25,
            "UIUC05_wood2": 26,
            "UIUC14_brick1": 27,
            "UIUC19_carpet2": 28,
            "UIUC23_knit": 29,
            "UIUC22_fur": 30,
            "UIUC15_brick2": 31,
            "KTH_wool": 32,
            "KTH_orange_peel": 33,
            "Kyberge_blanket2": 34,
            "Kyberge_sand1": 35,
            "KTH_sponge": 36,
            "Kyberge_seat1": 37,
            "Kyberge_scarf2": 38,
            "KTH_cracker": 39,
            "Kyberge_grass1": 40,
            "Kyberge_rice1": 41,
            "KTH_cork": 42,
            "UIUC04_wood1": 43,
            "Kyberge_cushion1": 44,
            "Kyberge_stone3": 45,
            "UIUC18_carpet1": 46,
            "Kyberge_ceiling2": 47,
            "UIUC10_floor1": 48,
            "Kyberge_floor1": 49,
            "Kyberge_stone2": 50,
            "KTH_cotton": 51,
            "UIUC09_marble": 52,
            "Kyberge_wall1": 53,
            "Kyberge_linseeds1": 54,
            "UIUC12_pebbles": 55,
            "UIUC11_floor2": 56,
            "UIUC01_bark1": 57,
            "Kyberge_rug1": 58,
            "KTH_styrofoam": 59,
            "UIUC25_plaid": 60,
            "UIUC21_wallpaper": 61,
            "UIUC13_wall": 62,
            "UIUC20_upholstery": 63,

        }

        self.opt = {
            "data_split_mode": "random",
            "classifier_opt": {
                "aggregator": "bag_of_words",
                "complexity": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                "type": "svm",
                # "n_estimators": 5000,
                # "param_grid": pg.support_vector_machine_grid(),
            },
            "feature": ["hsv-lbp"],
            "sampling_method": "dense",
            "sampling_step": 8,
            "sampling_window": 8,
            "image_size": {
                # "roi": [0.35, 0.5, 0.5, 0.99],
                "width": 256,
                "height": 256,
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
