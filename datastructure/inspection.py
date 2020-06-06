import os
import numpy as np
import cv2
from multiprocessing import Pool
from utils.label_file_utils import read_xml_file, read_bboxes_file, read_classification_label_file

from datastructure.box_tag import BoxTag

from utils.utils import check_n_make_dir


class Inspection:
    def __init__(self, path_to_inspection_data, tag_type, class_mapping):
        self.path_to_data = path_to_inspection_data
        self.tag_type = tag_type
        self.class_mapping = class_mapping
        self.images_folder = os.path.join(path_to_inspection_data, 'images')
        self.labels_folder = os.path.join(path_to_inspection_data, 'labels')
        self.tags_folder = os.path.join(path_to_inspection_data, 'tags')
        self.processed_images_folder = os.path.join(path_to_inspection_data, 'processed_img')

        self.tags = None

        self.load_all = False

        self.inspection_information = dict()
        self.load_additional_information()

    def get_inspection_time_stamp(self):
        if self.tags is None:
            print("Warning no Time Stamp could be generated")
            return None
        else:
            time_stamp = []
            for tag_id in self.tags:
                tag = self.tags[tag_id]
                time_stamp.append(tag.time_stamp)
            return np.mean(time_stamp)

    def load_tags(self, load_all=False):
        self.load_all = load_all
        if self.tag_type == "box":
            self._load_box_tags()
        elif self.tag_type == "cls":
            self._load_cls_tags()
        else:
            raise ValueError("Tag Type {} is not recognised!".format(self.tag_type))

    def load_additional_information(self):
        path_to_info = os.path.join(self.path_to_data, "additional_information.txt")
        if os.path.isfile(path_to_info):
            with open(path_to_info) as info_f:
                for line in info_f:
                    target, information = line.strip().split(",")
                    self.inspection_information[target] = information

    def _load_box_tags(self):
        print("Collecting Box Tags...")
        self.tags = dict()
        pool = Pool()
        tags = pool.map(self.load_single_box_tag,  sorted(os.listdir(self.labels_folder)))
        pool.close()
        pool.join()
        for t in tags:
            if t is not None and t.is_valid():
                self.tags[t.tag_id] = t
        print("Box tags collected.")

    def _load_cls_tags(self):
        print("Collecting images as tags")
        self.tags = dict()
        pool = Pool()
        tags = pool.map(self.load_single_cls_tag, sorted(os.listdir(self.labels_folder)))
        pool.close()
        pool.join()
        for t in tags:
            if t is not None and t.is_valid():
                self.tags[t.tag_id] = t
        print("Cls tags collected.")

    def get_tags(self, classes_to_consider="all"):
        tags_out = dict()
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            if tag.has_relevant_classes(classes_to_consider) or classes_to_consider == "all":
                tag.information = self.inspection_information
                tags_out[tag_id] = tag
        return tags_out

    def export_tags(self):
        check_n_make_dir(self.tags_folder)
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            tag.export(self.tags_folder)

    def load_single_box_tag(self, lb_f):
        bboxes = []
        tag = None
        img_id = None
        if lb_f.endswith(".xml"):
            img_id = lb_f.replace(".xml", ".jpg")
            img_id = os.path.join(self.images_folder, img_id)
            bboxes = read_xml_file(os.path.join(self.labels_folder, lb_f))
        if lb_f.endswith(".txt"):
            img_id = lb_f.replace(".txt", ".jpg")
            img_id = os.path.join(self.images_folder, img_id)
            bboxes = read_bboxes_file(os.path.join(self.labels_folder, lb_f), img_id)

        if bboxes is not None:
            for idx, box in enumerate(bboxes):
                if box[0] in self.class_mapping:
                    tag_id = lb_f
                    tag = BoxTag(tag_id, img_id, [box[0]], box, self.class_mapping)
                else:
                    if self.load_all:
                        tag_id = lb_f
                        tag = BoxTag(tag_id, img_id, ["bg"], box, self.class_mapping)
        return tag

    def load_single_cls_tag(self, lb_f):
        tag = None
        if lb_f.endswith(".txt"):
            img_id = lb_f.replace(".txt", ".jpg")
            img_id = os.path.join(self.images_folder, img_id)
            if not os.path.isfile(img_id):
                img_id = lb_f.replace(".txt", ".ppm")
                img_id = os.path.join(self.images_folder, img_id)
            if not os.path.isfile(img_id):
                img_id = lb_f.replace(".txt", ".tif")
                img_id = os.path.join(self.images_folder, img_id)
            tag_id = img_id
            class_names = read_classification_label_file(os.path.join(self.labels_folder, lb_f))
            img = cv2.imread(img_id)
            if img is None:
                return None
            height, width = img.shape[:2]
            load = False
            for class_name in class_names:
                if class_name in self.class_mapping:
                    load = True

            if load or self.load_all:
                box = [class_names, 0, 0, width - 1, height - 1]
                tag = BoxTag(tag_id, img_id, class_names, box, self.class_mapping)

                return tag

        return tag
