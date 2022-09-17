import argparse
import os
import numpy as np
from tqdm import tqdm
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification.machine_learning import ClassicImageClassifier
from classic_image_classification.utils.utils import check_n_make_dir

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, classification_report


from classic_image_classification.utils.outlier_removal import get_best_threshold


def make_displot(data_frame, key, model_folder):
    sns.displot(data=data_frame, x=key, hue="status", kind="kde")
    plt.savefig(os.path.join(model_folder, "{}.png".format(key)))
    plt.close()


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    dt = args_.dataset_type

    model = ClassicImageClassifier()
    model.load(mf)

    d_set = DataSet(
        data_set_dir=df,
        tag_type=dt,
        class_mapping=model.class_mapping
    )
    d_set.load_data()
    tag_set = d_set.get_tags(classes_to_consider="all")

    results_to_plot_later = []

    print("[INFO] Analysis of Tags ...")
    data_frame = []
    for tag_id in tqdm(tag_set):
        tag = tag_set[tag_id]
        cls_id, confidence = model.predict_image(tag.load_data(), get_confidence=True)
        print(cls_id, confidence)
        t_class_name = tag.tag_class[0]
        if t_class_name == "bg":
            status = "unknown"
            status_id = 0
        else:
            status = "known"
            status_id = 1

        pred_cls_name = model.class_mapping_inv[cls_id]
        results_to_plot_later.append([tag, confidence, status, pred_cls_name])
        data_frame.append({
            "class_name": t_class_name,
            "pred_class_name": pred_cls_name,
            "status": status,
            "status_id": status_id,
            "max_score": confidence,

        })

    data_frame = pd.DataFrame(data_frame)
    print(data_frame)

    auroc = roc_auc_score(np.array(data_frame["status_id"]), np.array(data_frame["max_score"]))
    best_threshold = get_best_threshold(np.array(data_frame["max_score"]), np.array(data_frame["status_id"]))
    print("[RESULT] AUROC: {}".format(auroc))
    print("[RESULT] Threshold: {}".format(best_threshold))
    score = np.array(data_frame["max_score"])
    prediction = np.zeros(score.shape)
    prediction[score >= best_threshold] = 1
    print(classification_report(
        np.array(data_frame["status_id"]),
        prediction
    ))

    with open(os.path.join(mf, "outlier_score.txt"), "w") as f:
        f.write("[RESULT] AUROC: {}\n [RESULT] Threshold: {}".format(auroc, best_threshold))

    data_frame.to_csv(os.path.join(mf, "data_frame_outlier.csv"))

    for key in ["max_score"]:
        make_displot(data_frame, key, mf)

    print("[INFO] Plotting Results...")
    fp_outlier_dir = os.path.join(mf, "wrongly_accepted_outlier")
    fn_outlier_dir = os.path.join(mf, "wrongly_rejected_outlier")

    outlier_counts = {"status": [], "class_name": []}
    check_n_make_dir(fp_outlier_dir, clean=True)
    check_n_make_dir(fn_outlier_dir, clean=True)
    for tag, tag_score, tag_status, pred_cls_name in results_to_plot_later:
        if tag_score >= best_threshold and tag_status == "unknown":
            tag.export_box(os.path.join(fp_outlier_dir, pred_cls_name[:20]), with_text=True)
            outlier_counts["status"].append(tag_status)
            outlier_counts["class_name"].append(pred_cls_name[:20])
        if tag_score < best_threshold and tag_status == "known":
            tag.export_box(os.path.join(fn_outlier_dir, tag.tag_class[0][:20]), with_text=True)
            outlier_counts["status"].append(tag_status)
            outlier_counts["class_name"].append(tag.tag_class[0][:20])

    outlier_counts = pd.DataFrame(outlier_counts)
    for stat, stat_grp in outlier_counts.groupby("status"):
        print("STATUS: {}".format(stat))
        print(stat_grp["class_name"].value_counts())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with known data to fit the initial model",
    )
    parser.add_argument(
        "--model_folder",
        "-m",
        help="Path to save the model to",
    )
    parser.add_argument(
        "--test_data_folder",
        "-tf",
        default=None,
        help="Path to directory with known and unknown data to test on",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
