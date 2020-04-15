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


def show_results(result_dict):
    print(" ")
    print("Model Scores:")
    for cls in result_dict:
        tp = result_dict[cls]["tp"]
        fp = result_dict[cls]["fp"]
        fn = result_dict[cls]["fn"]

        eps = 1e-5

        pre = tp / (fp + tp + eps)
        rec = tp / (fn + tp + eps)
        acc = tp / (fn + fp + tp + eps)
        f_1 = 2 * (pre * rec) / (pre + rec + eps)

        print("------------")
        print("Class: {}".format(cls))
        print("Pre: {}".format(pre))
        print("Rec: {}".format(rec))
        print("Acc: {}".format(acc))
        print("F_1: {}".format(f_1))
        print("------------")


def save_results(result_path, pipeline_name, result_dict):
    result_str = ""
    result_str += "Model Scores: \n"
    for cls in result_dict:
        tp = result_dict[cls]["tp"]
        fp = result_dict[cls]["fp"]
        fn = result_dict[cls]["fn"]

        eps = 1e-5

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

    with open(os.path.join(result_path, "report_{}.txt".format(pipeline_name)), "w") as rep_f:
        rep_f.write(result_str)
    print("Model stats were saved to {}".format(os.path.join(result_path, "report_{}.txt".format(pipeline_name))))
