import cv2
import os

import numpy as np

from classic_image_classification import ImageEmbedding
import argparse


def build_result_image(image, list_of_tags):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    border = np.zeros((256, 10, 3))
    for tag in list_of_tags:
        data = tag.load_data()
        data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.concatenate([image, border, data], axis=1)
    return image


def main(args_):
    path = args_.dataset_folder
    opt = {
        "aggregator": "bag_of_words",
        "complexity": 128,
        "feature": "hsv-sift",
        "sampling_method": "dense",
        "sampling_step": 16,
        "sampling_window": 16,
        "image_size": {
            "width": 128,
            "height": 128,
        }
    }
    img_emb = ImageEmbedding(opt)

    img_emb.register_data_set(data_path=path, tag_type="cls")

    image = cv2.imread("/Users/fmuenke/datasets/abs-halteverbot/query.jpeg")

    res_match, res_mismatch = img_emb.query(image)

    matches = build_result_image(image, res_match)
    cv2.imwrite("/Users/fmuenke/datasets/abs-halteverbot/answer.jpeg", matches)

    mismatch = build_result_image(image, res_mismatch)
    cv2.imwrite("/Users/fmuenke/datasets/abs-halteverbot/answer_mismatch.jpeg", mismatch)


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
