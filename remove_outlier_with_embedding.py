import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from handcrafted_image_representations.machine_learning.outlier_detector import OutlierDetectorSearch, OutlierDetector


def make_displot(data_frame, key, model_folder):
    sns.displot(data=data_frame, x=key, hue="status", kind="kde")
    plt.savefig(os.path.join(model_folder, "{}.png".format(key)))
    plt.close()


class Config:
    def __init__(self, model_folder):
        self.down_sample = 0.0

        self.class_mapping = {
            "cls_0": 0,
            "cls_1": 1,
        }
        self.mf = model_folder

        self.opt = {
            "data_split_mode": "random",
            "method": "by_classifier",
            "aggregator": ["bag_of_words"],
            "complexity": [64, 128, 256, 512, 1024],
            "feature": ["gray-sift", "rgb-sift"],
            "sampling_method": "dense",
            "sampling_step": [16],
            "sampling_window": [16, 32],
            "image_size": [
                {
                    "width": 64,
                    "height": 64,
                },
            ]
        }


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    dt = args_.dataset_type
    tf = args_.test_data_folder

    cfg = Config(mf)

    od_search = OutlierDetectorSearch(cfg.opt, cfg.class_mapping)
    od_search.fit(mf, data_path_known=df, data_path_test=tf, tag_type=dt)

    od = OutlierDetector()
    od.load(mf)
    od.evaluate(tf, tag_type=dt, results_path=mf)


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
