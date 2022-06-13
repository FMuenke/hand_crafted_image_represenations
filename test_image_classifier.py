import argparse
import os
from classic_image_classification.machine_learning import ClassicImageClassifier


def test(mf, df, tc=None, dt="cls"):

    if tc is None:
        load_all = True
    else:
        load_all = False

    model = ClassicImageClassifier()
    model.load(mf)
    model.evaluate(df, dt, load_all, os.path.join(mf, "image_classification"))


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    tc = args_.target_class
    dt = args_.dataset_type
    test(mf, df, tc, dt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-m",
        default="./test/",
        help="Path to model",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    parser.add_argument(
        "--target_class",
        "-t",
        default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
