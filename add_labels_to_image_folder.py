import argparse
import os


def create_custom_labels(args_):
    images_dir = args_.images
    labels_dir = os.path.join(images_dir.replace("images", "labels"))

    label = args_.label

    print("Creating Labels")
    print(images_dir)
    print(labels_dir)

    if os.path.isdir(labels_dir):
        print("Error: There are already labels which should not be deleted!")
    else:
        os.mkdir(labels_dir)
        for img_f in os.listdir(images_dir):
            lb_f = img_f.replace(".jpg", ".txt")
            lb_f = lb_f.replace(".png", ".txt")
            lb_f = lb_f.replace(".ppm", ".txt")
            lb_f = lb_f.replace(".tif", ".txt")
            label_filepath = os.path.join(labels_dir, lb_f)
            s = ""
            for lab in label.split("/"):
                s += "{}\n".format(lab)
            with open(label_filepath, "w") as lb:
                lb.write(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", "-img", default="./images", help="Full path to directory with images to label"
    )
    parser.add_argument(
        "--label", "-l", help="Full path to directory with images to label"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_custom_labels(args)
