import pandas as pd
from classic_image_classification import ImageEmbedding
import argparse
import umap
import numpy as np
from classic_image_classification import DataSet
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    path = "/home/fmuenke/projects/sfm_object_matching/src/OpenSfM/data/reconstruction_project"
    opt = {
        "aggregator": "bag_of_words",
        "complexity": 128,
        "feature": "hsv-kaze",
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
    reducer.fit_transform(x)

    ds = DataSet(data_set_dir=path, tag_type="cls")
    ds.load_data()
    tags = ds.get_tags()
    print(len(tags))

    df = []
    for t_id in tags:
        x = img_emb.transform(tags[t_id].load_data())
        print(x)
        x = reducer.transform(x)
        print(x)
        df.append({
            "name": tags[t_id].tag_class[0],
            "f1": x[0], "f2": x[1],
        })

    df = pd.DataFrame(df)
    sns.scatterplot(data=df, x="f1", y="f2", hue="name")
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
