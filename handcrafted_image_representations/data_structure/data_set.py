import os
import logging
import imagesize
from handcrafted_image_representations.data_structure.box_tag import BoxTag
from handcrafted_image_representations.data_structure.labeled_image import LabeledImage

from multiprocessing import Pool


class DataSet:
    def __init__(self, data_set_dir, tag_type, class_mapping=None):
        self.tags = dict()
        self.data_set_dir = data_set_dir
        if class_mapping is None:
            class_mapping = {}
        self.class_mapping = class_mapping
        self.tag_type = tag_type

        self.load_data()

    def load_labeled_image(self, args):
        base_dir, img_f = args
        tags = []
        l_img = LabeledImage(base_dir, img_f[:-4])
        if l_img.image_file is None:
            return []
        if self.tag_type == "box":
            boxes = l_img.load_boxes()
            for box in boxes:
                t = BoxTag(
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

    def load_classes_directly_from_folders(self, path_to_directory):
        for sub_f in os.listdir(path_to_directory):
            if not os.path.isdir(os.path.join(path_to_directory, sub_f)):
                continue
            for img_f in os.listdir(os.path.join(path_to_directory, sub_f)):
                try:
                    image_file = os.path.join(path_to_directory, sub_f, img_f)
                    if img_f.endswith((".png", ".jpg")):
                        width, height = imagesize.get(image_file)
                        tag = BoxTag(
                            path_to_image=image_file,
                            tag_class=[sub_f],
                            box=[sub_f, 0, 0, width - 1, height - 1],
                            class_mapping=self.class_mapping
                        )
                        if tag.is_valid():
                            self.tags[len(self.tags)] = tag
                except Exception as e:
                    logging.error(e)
                    logging.error(img_f)

    def load_data(self):
        logging.info("[INFO] Loading data...")
        if self.tag_type == "folder":
            self.load_classes_directly_from_folders(self.data_set_dir)
        else:
            self.load_directory(self.data_set_dir)
            for d in os.listdir(self.data_set_dir):
                self.load_directory(os.path.join(self.data_set_dir, d))

    def get_tags(self, classes_to_consider="all"):
        tags_out = []
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            if tag.has_relevant_classes(classes_to_consider) or classes_to_consider == "all":
                tags_out.append(tag)
        logging.info("[INFO] {} instances were loaded.".format(len(tags_out)))
        return tags_out
