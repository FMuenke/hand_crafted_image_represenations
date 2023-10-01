import argparse
import os
from hand_crafted_image_representations.utils.utils import save_dict


def collect_labels_from_dir(df, list_of_labels):
    lb_dir = os.path.join(df, "labels")
    if os.path.isdir(lb_dir):
        for lb_file in os.listdir(lb_dir):
            lb_file_name = os.path.join(lb_dir, lb_file)
            if lb_file_name.endswith(".txt"):
                with open(lb_file_name) as lf:
                    for lab in lf:
                        lab = lab.replace("\n", "")
                        if lab not in list_of_labels:
                            list_of_labels.append(lab)
    return list_of_labels


def collect_labels(df):
    list_of_labels = []
    list_of_labels = collect_labels_from_dir(df, list_of_labels)
    for lb_df in os.listdir(df):
        path_to_df = os.path.join(df, lb_df)
        if os.path.isdir(path_to_df):
            list_of_labels = collect_labels_from_dir(path_to_df, list_of_labels)

    s = ""
    label_json = {}
    for i, label in enumerate(list_of_labels):
        s += "\"{}\": {},\n".format(label, i)
        label_json[label] = i

    with open(os.path.join(df, "label_list.txt"), "w") as f:
        f.write(s)

    save_dict(label_json, os.path.join(df, "class_mapping.json"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", "-df", default="./images", help="Full path to directory with images to label"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_labels(args.data_folder)
