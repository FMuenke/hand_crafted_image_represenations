import logging
from classic_image_classification import DataSet, ClassicImageClassifier


def main():
    split = 0.25

    class_mapping = {
        "manhole": 1,
        "stormdrain": 2,
        # "crack": 3,
    }

    opt = {
        "data_split_mode": "random",
        "data_split_ratio": 0.2,
        "aggregator": "bag_of_words",
        "complexity": 16,
        "type": "svm",
        # "n_estimators": 5000,
        # "param_grid": pg.support_vector_machine_grid(),
        "feature": "hsv-hog+8+L2",
        "sampling_method": "dense",
        "sampling_step": 16,
        "sampling_window": 16,
        "image_size": {
            "width": 128,
            "height": 128,
            "padding": False,
        },
    }


    mf = "/home/fmuenke/ai_models/emb_test_2"

    cls = ClassicImageClassifier(
        opt=opt,
        class_mapping=class_mapping)

    cls.fit("/home/fmuenke/datasets/manhole-stormdrain-defect-dataset/train", tag_type="cls")



if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
