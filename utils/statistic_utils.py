import os


def init_result_dict(class_mapping):
    result_dict = dict()
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
