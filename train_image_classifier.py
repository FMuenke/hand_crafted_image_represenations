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
            "n02090379-redbone": 0,
            "n02112137-chow": 1,
            "n02091831-Saluki": 2,
            "n02094258-Norwich_terrier": 3,
            "n02107142-Doberman": 4,
            "n02104029-kuvasz": 5,
            "n02095314-wire-haired_fox_terrier": 6,
            "n02086910-papillon": 7,
            "n02106550-Rottweiler": 8,
            "n02113624-toy_poodle": 9,
            "n02102040-English_springer": 10,
            "n02111889-Samoyed": 11,
            "n02110806-basenji": 12,
            "n02087394-Rhodesian_ridgeback": 13,
            "n02094433-Yorkshire_terrier": 14,
            "n02110063-malamute": 15,
            "n02100583-vizsla": 16,
            "n02115641-dingo": 17,
            "n02091467-Norwegian_elkhound": 18,
            "n02105056-groenendael": 19,
            "n02108915-French_bulldog": 20,
            "n02096051-Airedale": 21,
            "n02101556-clumber": 22,
            "n02093647-Bedlington_terrier": 23,
            "n02101388-Brittany_spaniel": 24,
            "n02086240-Shih-Tzu": 25,
            "n02106030-collie": 26,
            "n02106166-Border_collie": 27,
            "n02088094-Afghan_hound": 28,
            "n02105505-komondor": 29,
            "n02113186-Cardigan": 30,
            "n02085936-Maltese_dog": 31,
            "n02098286-West_Highland_white_terrier": 32,
            "n02107683-Bernese_mountain_dog": 33,
            "n02109525-Saint_Bernard": 34,
            "n02088364-beagle": 35,
            "n02112018-Pomeranian": 36,
            "n02086079-Pekinese": 37,
            "n02105412-kelpie": 38,
            "n02088238-basset": 39,
            "n02102177-Welsh_springer_spaniel": 40,
            "n02099267-flat-coated_retriever": 41,
            "n02105641-Old_English_sheepdog": 42,
            "n02106382-Bouvier_des_Flandres": 43,
            "n02097130-giant_schnauzer": 44,
            "n02102480-Sussex_spaniel": 45,
            "n02107574-Greater_Swiss_Mountain_dog": 46,
            "n02105855-Shetland_sheepdog": 47,
            "n02094114-Norfolk_terrier": 48,
            "n02115913-dhole": 49,
            "n02110958-pug": 50,
            "n02108000-EntleBucher": 51,
            "n02099429-curly-coated_retriever": 52,
            "n02113978-Mexican_hairless": 53,
            "n02090721-Irish_wolfhound": 54,
            "n02089078-black-and-tan_coonhound": 55,
            "n02089867-Walker_hound": 56,
            "n02107312-miniature_pinscher": 57,
            "n02105251-briard": 58,
            "n02112706-Brabancon_griffon": 59,
            "n02097298-Scotch_terrier": 60,
            "n02109961-Eskimo_dog": 61,
            "n02093991-Irish_terrier": 62,
            "n02112350-keeshond": 63,
            "n02090622-borzoi": 64,
            "n02105162-malinois": 65,
            "n02108422-bull_mastiff": 66,
            "n02093754-Border_terrier": 67,
            "n02107908-Appenzeller": 68,
            "n02108089-boxer": 69,
            "n02106662-German_shepherd": 70,
            "n02089973-English_foxhound": 71,
            "n02113023-Pembroke": 72,
            "n02097658-silky_terrier": 73,
            "n02116738-African_hunting_dog": 74,
            "n02096177-cairn": 75,
            "n02111500-Great_Pyrenees": 76,
            "n02093256-Staffordshire_bullterrier": 77,
            "n02093859-Kerry_blue_terrier": 78,
            "n02092339-Weimaraner": 79,
            "n02096294-Australian_terrier": 80,
            "n02108551-Tibetan_mastiff": 81,
            "n02095889-Sealyham_terrier": 82,
            "n02110185-Siberian_husky": 83,
            "n02097474-Tibetan_terrier": 84,
            "n02101006-Gordon_setter": 85,
            "n02100236-German_short-haired_pointer": 86,
            "n02085782-Japanese_spaniel": 87,
            "n02100735-English_setter": 88,
            "n02091134-whippet": 89,
            "n02099849-Chesapeake_Bay_retriever": 90,
            "n02098413-Lhasa": 91,
            "n02102973-Irish_water_spaniel": 92,
            "n02091032-Italian_greyhound": 93,
            "n02100877-Irish_setter": 94,
            "n02111129-Leonberg": 95,
            "n02113799-standard_poodle": 96,
            "n02099601-golden_retriever": 97,
            "n02113712-miniature_poodle": 98,
            "n02088466-bloodhound": 99,
            "n02097209-standard_schnauzer": 100,
            "n02098105-soft-coated_wheaten_terrier": 101,
            "n02092002-Scottish_deerhound": 102,
            "n02099712-Labrador_retriever": 103,
            "n02111277-Newfoundland": 104,
            "n02085620-Chihuahua": 105,
            "n02096585-Boston_bull": 106,
            "n02097047-miniature_schnauzer": 107,
            "n02096437-Dandie_Dinmont": 108,
            "n02091635-otterhound": 109,
            "n02095570-Lakeland_terrier": 110,
            "n02087046-toy_terrier": 111,
            "n02093428-American_Staffordshire_terrier": 112,
            "n02091244-Ibizan_hound": 113,
            "n02102318-cocker_spaniel": 114,
            "n02086646-Blenheim_spaniel": 115,
            "n02088632-bluetick": 116,
            "n02104365-schipperke": 117,
            "n02109047-Great_Dane": 118,
            "n02110627-affenpinscher": 119,
        }

        self.opt = {
            "data_split_mode": "random",
            "classifier_opt": {
                "aggregator": "bag_of_words",
                "complexity": [5, 10, 15, 25, 50, 75, 100, 250, 500, 750, 1000, 1250],
                "type": "b_rf",
                "n_estimators": 2500,
                # "param_grid": pg.support_vector_machine_grid(),
            },
            "feature": ["hsv-hog"],
            "sampling_method": "dense",
            "sampling_step": 16,
            "sampling_window": 16,
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

    agg_complexities = cfg.opt["classifier_opt"]["complexity"]
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
