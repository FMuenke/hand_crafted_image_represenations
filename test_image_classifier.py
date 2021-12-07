import argparse
import os
from tqdm import tqdm
from classic_image_classification.data_structure.data_set import DataSet
from classic_image_classification.machine_learning.ensemble_image_classifier import EnsembleImageClassifier
from classic_image_classification.utils.utils import check_n_make_dir
from classic_image_classification.utils.statistic_utils import init_result_dict, show_results, save_results


def test(mf, df, tc=None, dt="cls"):

    if tc is None:
        tc = "all"
    else:
        tc = [tc]

    prediction_folder = os.path.join(mf, "image_wise_classification")
    check_n_make_dir(prediction_folder, clean=True)

    ML_pipeline = EnsembleImageClassifier(mf)
    ML_pipeline.load()

    d_set = DataSet(data_set_dir=df, class_mapping=ML_pipeline.class_mapping, tag_type=dt)
    d_set.load_data()
    tag_set = d_set.get_tags(classes_to_consider=tc)

    print("Analysis of Tags ...")
    result_dict = init_result_dict(d_set.get_class_mapping())
    for tag_id in tqdm(tag_set):
        tag = d_set.tags[tag_id]
        y_pred = ML_pipeline.predict(tag)
        tag.write_prediction(y_pred, prediction_folder)
        result_dict = tag.evaluate_prediction(y_pred, result_dict)

    show_results(result_dict)
    save_results(prediction_folder, "image_classifier", result_dict)


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    tc = args_.target_class
    dt = args_.dataset_type
    test(mf, df, tc, dt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-m",
        default="./test/",
        help="Path to model",
    )
    parser.add_argument(
        "--dataset_type",
        "-dtype",
        default="cls",
        help="Choose Dataset Annotation Bounding-Boxes [box] or Image Labels [cls]",
    )
    parser.add_argument(
        "--target_class",
        "-t",
        default=None
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
