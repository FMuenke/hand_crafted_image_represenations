import argparse
import os

from utils.utils import save_dict
from utils.label_file_utils import read_classification_label_file


def parse_labels(path, all_labels):

    for l_f in os.listdir(path):
        if l_f.endswith(".txt"):
            label_list = read_classification_label_file(os.path.join(path, l_f))
            for lab in label_list:
                if lab not in all_labels:
                    all_labels[lab] = len(all_labels)
    return all_labels


def main(args_):
    df = args_.data_folder

    all_labels = {}

    labels_dir = os.path.join(df, "labels")
    if os.path.isdir(labels_dir):
        all_labels = parse_labels(labels_dir, all_labels)
    else:
        for d in os.listdir(df):
            labels_dir = os.path.join(df, d, "labels")
            if os.path.isdir(labels_dir):
                all_labels = parse_labels(labels_dir, all_labels)

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
