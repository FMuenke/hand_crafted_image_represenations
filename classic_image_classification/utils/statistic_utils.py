import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def init_result_dict(class_mapping):
    result_dict = dict()
    result_dict["overall"] = {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
    for cls in class_mapping:
        result_dict[cls] = {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    return result_dict


def make_string(result_dict):
    result_str = ""
    result_str += "Model Scores: \n"
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    eps = 1e-5
    for cls in result_dict:
        tp = result_dict[cls]["tp"]
        fp = result_dict[cls]["fp"]
        fn = result_dict[cls]["fn"]
        tot_fp += fp
        tot_fn += fn
        tot_tp += tp

        pre = tp / (fp + tp + eps)
        rec = tp / (fn + tp + eps)
        acc = tp / (fn + fp + tp + eps)
        f_1 = 2 * (pre * rec) / (pre + rec + eps)

        result_str += "------------\n"
        result_str += "Class: {}\n".format(cls)
        result_str += "Pre: {}\n".format(pre)
        result_str += "Rec: {}\n".format(rec)
        result_str += "Acc: {}\n".format(acc)
        result_str += "F_1: {}\n".format(f_1)
        result_str += "------------\n"

    pre = tot_tp / (tot_fp + tot_tp + eps)
    rec = tot_tp / (tot_fn + tot_tp + eps)
    acc = tot_tp / (tot_fn + tot_fp + tot_tp + eps)
    f_1 = 2 * (pre * rec) / (pre + rec + eps)

    result_str += "------------\n"
    result_str += "Class: {}\n".format("overall")
    result_str += "Pre: {}\n".format(pre)
    result_str += "Rec: {}\n".format(rec)
    result_str += "Acc: {}\n".format(acc)
    result_str += "F_1: {}\n".format(f_1)
    result_str += "------------\n"
    return result_str


def show_results(result_dict):
    print(make_string(result_dict))


def save_results(result_path, pipeline_name, result_dict):
    result_str = make_string(result_dict)

    with open(os.path.join(result_path, "report_{}.txt".format(pipeline_name)), "w") as rep_f:
        rep_f.write(result_str)
    print("Model stats were saved to {}".format(os.path.join(result_path, "report_{}.txt".format(pipeline_name))))


def plot_roc(y, predictions, confidences, class_mapping, mf):
    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    conf_thresholds = np.arange(0, 0.99, 0.005)

    y = np.array(y)
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    results = []

    unique_classes = np.unique(y)
    for uy in unique_classes:
        y_sc = np.zeros(y.shape)
        y_sc[predictions == uy] = confidences[predictions == uy]
        y_sc[predictions != uy] = 1 - confidences[predictions != uy]
        for c in conf_thresholds:
            tp_ids = np.where(np.logical_and(y == uy, y_sc >= c))[0]
            fn_ids = np.where(np.logical_and(y == uy, y_sc < c))[0]
            fp_ids = np.where(np.logical_and(y != uy, y_sc >= c))[0]
            tp = len(tp_ids)
            fp = len(fp_ids)
            fn = len(fn_ids)
            pre = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            results.append({
                "cls": class_mapping_inv[uy],
                "precision": pre,
                "recall": rec,
                "f1-score": 2 * rec * pre / (pre + rec + 1e-6),
                "confidence": c,
            })

    df = pd.DataFrame(results)
    plt.subplot(311)
    plt.title("CONFIDENCE - CHARACTERISTIC")
    sns.lineplot(data=df, x="confidence", y="precision", hue="cls")
    plt.subplot(312)
    sns.lineplot(data=df, x="confidence", y="recall", hue="cls")
    plt.subplot(313)
    sns.lineplot(data=df, x="confidence", y="f1-score", hue="cls")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(mf, "confidence_characteristic.png"))
    plt.show()

    precision_report = ""
    for cls, cls_group in df.groupby("cls"):
        precision_report += str(cls) + "\n"
        max_pre = cls_group[cls_group["precision"] == cls_group["precision"].max()]
        precision_report += "Precision: {} , Recall: {} , F1-Score: {}\n".format(
            max_pre["precision"].max(), max_pre["recall"].max(), max_pre["f1-score"].max()
        )

    if mf is not None:
        with open(os.path.join(mf, "max_precision_stats.txt"), "w") as f:
            f.write(precision_report)

    print(precision_report)
