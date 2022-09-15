import logging
from classic_image_classification import ClassicImageClassifier


def main():

    class_mapping = {
        "cobblestone": 0,
        "bg": 1,
    }

    opt = {
        "data_split_mode": "random",
        "data_split_ratio": 0.2,
        "aggregator": "bag_of_words",
        "complexity": 32,
        "type": "svm",
        # "n_estimators": 5000,
        # "param_grid": pg.support_vector_machine_grid(),
        "feature": "hsv-hog",
        "sampling_method": "dense",
        "sampling_step": 16,
        "sampling_window": 16,
        "image_size": {
            "width": 64,
            "height": 64,
        },
    }


    mf = "/home/fmuenke/ai_models/traffic_sign_test"

    cls = ClassicImageClassifier(
        opt=opt,
        class_mapping=class_mapping)

    cls.fit("/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/traffic_sign_classification/train_small", tag_type="cls")
    cls.save(mf)
    cls.evaluate("/media/fmuenke/8c63b673-ade7-4948-91ca-aba40636c42c/datasets/traffic_sign_classification/test",
                 tag_type="cls",
                 report_path=mf)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()