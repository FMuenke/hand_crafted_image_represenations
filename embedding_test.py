import pandas as pd
import os
from classic_image_classification import ImageEmbedding
import argparse
import cv2
from tqdm import tqdm
from umap import UMAP
import numpy as np
from classic_image_classification import DataSet
import seaborn as sns
import matplotlib.pyplot as plt

from classic_image_classification.utils.utils import check_n_make_dir


def main(args_):
    path = args_.dataset_folder
    opt = {
        "aggregator": "bag_of_words",
        "complexity": 128,
        "feature": "hsv-hog",
        "sampling_method": "dense",
        "sampling_step": 4,
        "sampling_window": 16,
        "image_size": {
            "width": 32,
            "height": 32,
        }
    }
    img_emb = ImageEmbedding(opt)
    x = img_emb.fit_transform(data_path=path, tag_type="cls")
    x = np.concatenate(x, axis=0)

    projection = UMAP(n_components=4, n_jobs=-1)
    x_proj = projection.fit_transform(x)

    ds = DataSet(data_set_dir=path, tag_type="cls")
    ds.load_data()
    tags = ds.get_tags()

    df = {"name": []}
    for t_id in tqdm(tags):
        df["name"].append(tags[t_id].tag_class[0])

    list_of_variables = []
    for i in range(x_proj.shape[1]):
        list_of_variables.append("x{}".format(i+1))
        df["x{}".format(i+1)] = x_proj[:, i]

    df = pd.DataFrame(df)
    sns.pairplot(data=df, vars=list_of_variables, hue="name", kind="kde")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with images to cluster",
    )
    parser.add_argument(
        "--result_folder",
        "-rf",
        help="Path to store results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
