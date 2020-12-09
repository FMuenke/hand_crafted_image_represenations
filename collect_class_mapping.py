import argparse
import os

from utils.utils import save_dict
from utils.label_file_utils import read_classification_label_file


def main(args_):
    df = args_.data_folder

    all_labels = {}

    label_folder = df
    if os.path.isdir(os.path.join(df, "labels")):
        label_folder = os.path.join(df, "labels")

    for l_f in os.listdir(label_folder):
        if l_f.endswith(".txt"):
            label_list = read_classification_label_file(os.path.join(label_folder, l_f))
            for lab in label_list:
                if lab not in all_labels:
                    all_labels[lab] = len(all_labels)

    save_dict(all_labels, os.path.join(df, "proto_class_mapping.json"))
    print("Classes successfully parsed: {} classes found.".format(len(all_labels)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", "-df", default="./images", help="Full path to directory with images to label"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
