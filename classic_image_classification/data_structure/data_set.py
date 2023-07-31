import os
import logging
from classic_image_classification.data_structure.box_tag import BoxTag
from classic_image_classification.data_structure.labeled_image import LabeledImage

from multiprocessing import Pool


class DataSet:
    def __init__(self, data_set_dir, tag_type, class_mapping=None):
        self.tags = dict()
        self.data_set_dir = data_set_dir
        self.class_mapping = class_mapping
        self.tag_type = tag_type

    def load_labeled_image(self, args):
        base_dir, img_f = args
        tags = []
        l_img = LabeledImage(base_dir, img_f[:-4])
        if l_img.image_file is None or l_img.label_file is None:
            return []
        if self.tag_type == "box":
            boxes = l_img.load_boxes()
            for box in boxes:
                t = BoxTag(
                    tag_id=len(self.tags),
                    path_to_image=l_img.image_file,
                    tag_class=[box[0]],
                    box=box,
                    class_mapping=self.class_mapping
                )
                if t.is_valid():
                    tags.append(t)
        elif self.tag_type == "cls":
            class_names = l_img.load_classes()
            height, width = l_img.get_image_size()
            box = [class_names, 0, 0, width - 1, height - 1]
            tags.append(BoxTag(
                tag_id=len(self.tags),
                path_to_image=l_img.image_file,
                tag_class=class_names,
                box=box,
                class_mapping=self.class_mapping
            ))
        else:
            raise Exception("UNKNOWN DATA MODE - {} -".format(self.tag_type))
        return tags

    def load_directory(self, path_to_directory):
        logging.info("[INFO] Loading Tags from: {}".format(path_to_directory))
        image_path = os.path.join(path_to_directory, "images")
        if os.path.isdir(os.path.join(path_to_directory, "images")):
            args = [[path_to_directory, img_f] for img_f in os.listdir(image_path)]
            pool = Pool()
            list_of_tag_sets = pool.map(self.load_labeled_image, args)
            pool.close()
            pool.join()
            for t_set in list_of_tag_sets:
                for t in t_set:
                    self.tags[len(self.tags)] = t

    def load_data(self):
        print("[INFO] Loading data...")
        self.load_directory(self.data_set_dir)
        for d in os.listdir(self.data_set_dir):
            self.load_directory(os.path.join(self.data_set_dir, d))

    def get_tags(self, classes_to_consider="all"):
        tags_out = dict()
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            if tag.has_relevant_classes(classes_to_consider) or classes_to_consider == "all":
                tags_out[tag_id] = tag
        print("[INFO] {} instances were loaded.".format(len(tags_out)))
        return tags_out
