import argparse
import os
from handcrafted_image_representations.machine_learning import ClassicImageClassifier


def test(mf, df, load_all, dt="cls"):

    model = ClassicImageClassifier()
    model.load(mf)
    model.evaluate(df, dt, load_all, os.path.join(mf, "image_classification"))


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    load_background = args_.load_background
    dt = args_.dataset_type
    test(mf, df, load_background, dt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-m",
        help="Path to model",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    parser.add_argument(
        "--load_background",
        "-bg",
        default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
