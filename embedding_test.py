import pandas as pd
import os
from classic_image_classification import ImageEmbedding
import argparse
import umap
import cv2
from sklearn.cluster import KMeans
import numpy as np
from classic_image_classification import DataSet
import seaborn as sns
import matplotlib.pyplot as plt

from classic_image_classification.utils.utils import check_n_make_dir


def main():
    path = "/home/fmuenke/projects/sfm_object_matching/src/OpenSfM/data/reconstruction_project"
    result_path = "/home/fmuenke/datasets/ts_clustering"
    check_n_make_dir(result_path)
    opt = {
        "aggregator": "bag_of_words",
        "complexity": 32,
        "feature": "hsv-hog",
        "sampling_method": "dense",
        "sampling_step": 16,
        "sampling_window": 16,
        "image_size": {
            "width": 64,
            "height": 64,
        }
    }
    img_emb = ImageEmbedding(opt)
    x = img_emb.fit_transform(data_path=path, tag_type="cls")
    x = np.array(x)
    x = x[:, 0, :]
    print(x.shape)

    reducer = umap.UMAP(n_components=2)
    reducer.fit(x)

    clustering = KMeans()
    clustering.fit(x)

    ds = DataSet(data_set_dir=path, tag_type="cls")
    ds.load_data()
    tags = ds.get_tags()
    print(len(tags))

    df = []
    for t_id in tags:
        data = tags[t_id].load_data()
        x = img_emb.transform(data)
        x = x[0]
        x_red = reducer.transform(x)
        x_red = x_red[0]

        print(x)
        x_cl = clustering.predict(x)
        cluster_path = os.path.join(result_path, str(x_cl[0]))
        check_n_make_dir(cluster_path)
        print(x_cl)
        df.append({
            "name": tags[t_id].tag_class[0],
            "f1": x_red[0], "f2": x_red[1],
            "cluster": x_cl[0],
        })
        img_path = os.path.join(cluster_path, str(t_id) + "-" + tags[t_id].tag_class[0] + ".jpg")
        print(img_path)
        cv2.imwrite(img_path, data)

    df = pd.DataFrame(df)
    print(df)
    sns.scatterplot(data=df, x="f1", y="f2", hue="cluster", style="name")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )

if __name__ == "__main__":
    main()
