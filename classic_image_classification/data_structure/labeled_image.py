import os
import xml.etree.ElementTree as ET
from PIL import Image


def get_file_name(base_path, data_id, extensions):
    for ext in extensions:
        filename = os.path.join(base_path, data_id + ext)
        if os.path.isfile(filename):
            return filename
    return None


def read_xml_file(path_to_src_file, path_to_img_file):
    bboxes = []

    tree = ET.parse(path_to_src_file)
    root = tree.getroot()
    for child in root.findall("object"):
        name = child.find("name").text
        x1 = child.find("bndbox").find("xmin").text
        y1 = child.find("bndbox").find("ymin").text
        x2 = child.find("bndbox").find("xmax").text
        y2 = child.find("bndbox").find("ymax").text
        bboxes.append([name, float(x1), float(y1), float(x2), float(y2), 1.0])
    return bboxes


def read_txt_file(path_to_src_file, path_to_img_file):
    bboxes = []
    with open(path_to_src_file, "r") as src_f:
        for line in src_f:
            # In case there is no confidence value saved e.g. tamara tagging
            if len(line.strip().split(" ")) == 6:
                (class_name, x1_prz, y1_prz, x2_prz, y2_prz, prob) = line.strip().split(" ")
            elif len(line.strip().split(" ")) == 5:
                (class_name, x1_prz, y1_prz, x2_prz, y2_prz) = line.strip().split(" ")
                prob = 1.0
            else:
                continue

            im = Image.open(path_to_img_file)
            width, height = im.size
            x1_prz = float(x1_prz) * width
            y1_prz = float(y1_prz) * height
            x2_prz = float(x2_prz) * width
            y2_prz = float(y2_prz) * height

            prob = float(prob)
            bboxes.append([class_name, x1_prz, y1_prz, x2_prz, y2_prz, prob])
    return bboxes


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


class LabeledImage:

    image_extensions = [".jpg", ".JPG", ".png", "PNG", ".jpeg"]
    label_extensions = [".txt", ".xml"]

    def __init__(self, base_path, data_id):
        self.id = data_id
        self.path = base_path

        self.image_path = os.path.join(base_path, "images")
        self.label_path = os.path.join(base_path, "labels")

        self.image_file = get_file_name(self.image_path, self.id, self.image_extensions)
        self.label_file = get_file_name(self.label_path, self.id, self.label_extensions)

    def get_image_size(self):
        if self.image_file is None:
            raise Exception("NO IMAGE FILE AVAILABLE")
        im = Image.open(self.image_file)
        width, height = im.size
        return height, width

    def load_boxes(self):
        if self.image_file is None:
            return
        if self.label_file is None:
            return

        if self.label_file.endswith(".txt"):
            return read_txt_file(self.label_file, self.image_file)
        elif self.label_file.endswith(".xml"):
            return read_xml_file(self.label_file, self.image_file)
        else:
            raise Exception("UNEXPECTED EXTENSION: {}".format(self.label_file))

    def load_classes(self):
        if self.label_file.endswith(".txt"):
            return read_classification_label_file(self.label_file)
        else:
            raise Exception("UNEXPECTED EXTENSION: {}".format(self.label_file))


