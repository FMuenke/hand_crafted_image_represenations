import argparse
import os

from utils.utils import check_n_make_dir
import hjson

from shutil import copyfile


def main(args_):
    df = args_.data_folder

    with open(os.path.join(df, "dataset_split_0.hjson")) as infile:
        split = hjson.load(infile)

    be = os.path.join(df, "benigne_maligne", "benigne")
    ma = os.path.join(df, "benigne_maligne", "maligne")

    tr = os.path.join(df, "train")
    te = os.path.join(df, "test")
    check_n_make_dir(tr)
    check_n_make_dir(te)

    tr_ma = os.path.join(df, "train", "maligne")
    tr_be = os.path.join(df, "train", "benigne")
    te_ma = os.path.join(df, "test", "maligne")
    te_be = os.path.join(df, "test", "benigne")
    check_n_make_dir(tr_ma)
    check_n_make_dir(tr_be)
    check_n_make_dir(te_ma)
    check_n_make_dir(te_be)

    tr_ma = os.path.join(df, "train", "maligne", "images")
    tr_be = os.path.join(df, "train", "benigne", "images")
    te_ma = os.path.join(df, "test", "maligne", "images")
    te_be = os.path.join(df, "test", "benigne", "images")
    check_n_make_dir(tr_ma)
    check_n_make_dir(tr_be)
    check_n_make_dir(te_ma)
    check_n_make_dir(te_be)

    for im_f in os.listdir(be):
        src = os.path.join(be, im_f)
        if im_f in split["train"] or im_f in split["val"]:
            dst = os.path.join(tr_be, im_f)
            copyfile(src, dst)
        if im_f in split["test"]:
            dst = os.path.join(te_be, im_f)
            copyfile(src, dst)

    for im_f in os.listdir(ma):
        src = os.path.join(ma, im_f)
        if im_f in split["train"] or im_f in split["val"]:
            dst = os.path.join(tr_ma, im_f)
            copyfile(src, dst)
        if im_f in split["test"]:
            dst = os.path.join(te_ma, im_f)
            copyfile(src, dst)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", "-df", default="./images", help="Full path to directory with images to label"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
