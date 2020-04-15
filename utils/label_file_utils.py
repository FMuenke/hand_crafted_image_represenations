import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from PIL import Image


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def check_src_file_type(path_to_src_file: str):
    """
    Used to determine the type of input source [cls, box, lbm, unknown]
    To process it accordingly
    Args:
        path_to_src_file (str): path to source file to be classified

    Returns:
        the type of given source file [cls, box, lbm, unknown]
    """
    if path_to_src_file.endswith(".jpg"):
        mode = "lbm"
    elif path_to_src_file.endswith(".png"):
        mode = "lbm"
    elif path_to_src_file.endswith(".txt"):
        with open(path_to_src_file) as src_f:
            mode = "unknown"
            for line in src_f:
                if (
                    len(line.strip().split(" ")) == 6
                    or len(line.strip().split(" ")) == 5
                ):
                    mode = "box"

                if len(line.strip().split(" ")) == 1:
                    mode = "cls"

    else:
        print("Unexpected source file format: {}".format(path_to_src_file))
        mode = "unknown"

    return mode


def check_src_type(path_to_src_folder: str, idx=0):
    src_file = os.listdir(path_to_src_folder)[idx]
    path_to_src_file = os.path.join(path_to_src_folder, src_file)
    mode = check_src_file_type(path_to_src_file)

    if mode == "unknown":
        mode = check_src_type(path_to_src_folder, idx=idx+1)

    return mode


def translate_regression_to_classification(y_pred):
    y_out = []
    for y in y_pred:
        if y < 1.666:
            y = 1
        elif y > 2.666:
            y = 3
        else:
            y = 2
        y_out.append(y)
    return y_out


def read_classification_label_file(path_to_label_file):
    cls_list = []
    with open(path_to_label_file) as f:
        for line in f:
            info = line.split(" ")
            cls_list.append(info[0].replace("\n", ""))
    if len(cls_list) == 0:
        return ["bg"]
    else:
        return cls_list


def read_bboxes_file_soulaymen_format(path_to_src_file, path_to_img_file, absolute=True):
    """
    Used to read a prediction text file to bounding-box list (bboxes)
    Args:
        path_to_src_file: path to source txt file

    Returns:
        bboxes (list): [bbox, bbox2,...] bbox (list): [class_name, x1, y1, x2, y2, prob]

    """
    bboxes = []
    with open(path_to_src_file, "r") as src_f:
        for line in src_f:
            # In case there is no confidence value saved e.g. tamara tagging
            if len(line.strip().split(" ")) == 6:
                (x1_prz, y1_prz, x2_prz, y2_prz, prob, class_name) = line.strip().split(
                    " "
                )
            elif len(line.strip().split(" ")) == 5:
                (x1_prz, y1_prz, x2_prz, y2_prz, class_name) = line.strip().split(" ")
                prob = 1.0
            else:
                return None

            x1_prz = float(x1_prz)
            y1_prz = float(y1_prz)
            x2_prz = float(x2_prz)
            y2_prz = float(y2_prz)
            prz_format = True
            if x1_prz > 2:
                prz_format = False
            if x2_prz > 2:
                prz_format = False
            if y1_prz > 2:
                prz_format = False
            if y2_prz > 2:
                prz_format = False

            if prz_format:
                im = Image.open(path_to_img_file)
                width, height = im.size
                x1_prz = x1_prz*width
                y1_prz = y1_prz*height
                x2_prz = x2_prz*width
                y2_prz = y2_prz*height

            if not absolute:
                im = Image.open(path_to_img_file)
                width, height = im.size
                x1_prz = x1_prz / width
                y1_prz = y1_prz / height
                x2_prz = x2_prz / width
                y2_prz = y2_prz / height

            prob = float(prob)
            bboxes.append([class_name, x1_prz, y1_prz, x2_prz, y2_prz, prob])
    return bboxes


def read_bboxes_file(path_to_src_file, path_to_img_file, absolute=True):
    """
    Used to read a prediction text file to bounding-box list (bboxes)
    Args:
        path_to_src_file: path to source txt file

    Returns:
        bboxes (list): [bbox, bbox2,...] bbox (list): [class_name, x1, y1, x2, y2, prob]

    """
    bboxes = []
    with open(path_to_src_file, "r") as src_f:
        for line in src_f:
            # In case there is no confidence value saved e.g. tamara tagging
            if len(line.strip().split(" ")) == 6:
                (class_name, x1_prz, y1_prz, x2_prz, y2_prz, prob) = line.strip().split(
                    " "
                )
            elif len(line.strip().split(" ")) == 5:
                (class_name, x1_prz, y1_prz, x2_prz, y2_prz) = line.strip().split(" ")
                prob = 1.0
            else:
                return None

            x1_prz = float(x1_prz)
            y1_prz = float(y1_prz)
            x2_prz = float(x2_prz)
            y2_prz = float(y2_prz)
            prz_format = True
            if x1_prz > 2:
                prz_format = False
            if x2_prz > 2:
                prz_format = False
            if y1_prz > 2:
                prz_format = False
            if y2_prz > 2:
                prz_format = False

            if prz_format:
                im = Image.open(path_to_img_file)
                width, height = im.size
                x1_prz = x1_prz*width
                y1_prz = y1_prz*height
                x2_prz = x2_prz*width
                y2_prz = y2_prz*height

            if not absolute:
                im = Image.open(path_to_img_file)
                width, height = im.size
                x1_prz = x1_prz / width
                y1_prz = y1_prz / height
                x2_prz = x2_prz / width
                y2_prz = y2_prz / height

            prob = float(prob)
            bboxes.append([class_name, x1_prz, y1_prz, x2_prz, y2_prz, prob])
    return bboxes


def write_bboxes_file(path_to_src_file, bboxes):
    """
    Used to write bboxes to a prediction text file
    Args:
        path_to_src_file: path to text txt file
        bboxes (list): [bbox, bbox2,...] bbox (list): [class_name, x1, y1, x2, y2, prob]

    Returns:

    """
    pred_string = ""
    for box in bboxes:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        pred_string += "{0} {1} {2} {3} {4}\n".format(box[0], x1, y1, x2, y2)
    with open(path_to_src_file, "w") as src_f:
        src_f.write(pred_string)


def read_xml_file(path_to_xml_file):
    bboxes = []

    tree = ET.parse(path_to_xml_file)
    root = tree.getroot()
    for child in root.findall("object"):
        name = child.find("name").text
        x1 = child.find("bndbox").find("xmin").text
        y1 = child.find("bndbox").find("ymin").text
        x2 = child.find("bndbox").find("xmax").text
        y2 = child.find("bndbox").find("ymax").text
        bboxes.append([name, float(x1), float(y1), float(x2), float(y2)])
    return bboxes


def load_label_map(path_to_label_map, height, width, class_encoding):
    y_img = np.zeros((height, width))
    if path_to_label_map is not None:
        if os.path.isfile(path_to_label_map):
            lbm = cv2.imread(path_to_label_map)
            lbm = cv2.resize(lbm, (width, height), interpolation=cv2.INTER_NEAREST)
            for cls_idx in class_encoding:
                for x in range(width):
                    for y in range(height):
                        if lbm[y, x, 0] == class_encoding[cls_idx][0] \
                                and lbm[y, x, 1] == class_encoding[cls_idx][1] \
                                and lbm[y, x, 2] == class_encoding[cls_idx][2]:
                            y_img[y, x] = cls_idx
    return y_img

