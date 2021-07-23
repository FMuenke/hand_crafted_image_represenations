import argparse
import os


def convert_to_trainable(path_to_image_folder):
    im_dir = os.path.join(path_to_image_folder, "images")
    lb_dir = os.path.join(path_to_image_folder, "labels")

    if not os.path.isdir(lb_dir):
        os.mkdir(lb_dir)

    if not os.path.isdir(im_dir):
        os.mkdir(im_dir)

    label = os.path.basename(path_to_image_folder)
    label = label.replace("/", "")

    for f in os.listdir(path_to_image_folder):
        if f.endswith((".jpg", ".png", ".tif", ".ppm")):
            src_img_file = os.path.join(path_to_image_folder, f)
            dst_img_file = os.path.join(im_dir, f)
            os.rename(src_img_file, dst_img_file)
            with open(os.path.join(lb_dir, f[:-4] + ".txt"), "w") as lf:
                lf.write(label)

    return label


def main(args_):
    df = args_.dataset_folder
    label_list = []

    for d in os.listdir(df):
        if os.path.isdir(os.path.join(df, d)):
            print(d)
            label = convert_to_trainable(os.path.join(df, d))
            if label not in label_list:
                label_list.append(label)
    s = ""
    for i, label in enumerate(label_list):
        s += "\"{}\": {},\n".format(label, i)
    with open(os.path.join(df, "label_list.txt"), "w") as f:
        f.write(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
