import os

from datastructure.inspection import Inspection

from tqdm import tqdm

from utils.utils import check_n_make_dir


class DataSet:
    def __init__(self, data_set_dir, class_mapping):
        self.tags = dict()
        self.inspections = []
        self.data_set_dir = data_set_dir
        self.class_mapping = class_mapping

        self.current_idx = 0

        self.tags_folder = os.path.join(data_set_dir, "tags")

    def _add_tags(self, tag_dict):
        for tag_id in tag_dict:
            tag = tag_dict[tag_id]
            self.tags[len(self.tags)] = tag

    def _load_multiple_inspections(self, path_to_data_set, tag_type, load_all=False):
        for inspection_name in os.listdir(path_to_data_set):
            path_to_inspection = os.path.join(path_to_data_set, inspection_name)
            self._load_inspection(path_to_inspection, tag_type=tag_type, load_all=load_all)

    def _load_inspection(self, path_to_inspection, tag_type, load_all=False):
        if os.path.isdir(path_to_inspection):
            if os.path.isdir(os.path.join(path_to_inspection, "images")):
                print("Adding Inspection: {}".format(path_to_inspection))
                insp = Inspection(path_to_inspection,
                                  tag_type=tag_type,
                                  class_mapping=self.class_mapping)
                insp.load_tags(load_all=load_all)
                self._add_tags(insp.get_tags())
                if len(insp.tags) > 0:
                    self.inspections.append(path_to_inspection)

    def load_data(self, tag_type, load_all=False):
        if "bg" in self.class_mapping:
            load_all = True
        self._load_inspection(self.data_set_dir, tag_type, load_all=load_all)
        self._load_multiple_inspections(self.data_set_dir, tag_type, load_all=load_all)
        print("Found {} Tags.".format(len(self.tags)))

    def export_tags(self, classes_to_consider="all", surrounding=0):
        """
        Exporting cropped images of every tag.
        Tags are sorted by class and stored in image folders with matching label folders
        Returns:

        """
        check_n_make_dir(self.tags_folder, clean=True)
        print("Exporting Objects...")
        for tag_id in tqdm(self.tags):
            tag = self.tags[tag_id]
            if tag.has_relevant_classes(classes_to_consider) or classes_to_consider == "all":
                # Initialising dir where each class is stored
                tag.export(self.tags_folder, surrounding)
        print("Objects exported!")

    def get_class_mapping(self):
        return self.class_mapping

    def get_tags(self, inspection_id=None, classes_to_consider="all"):
        print("Exporting Tags...")
        tags_out = dict()
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            if tag.inspection_id == inspection_id or inspection_id is None:
                if tag.has_relevant_classes(classes_to_consider) or classes_to_consider == "all":
                    tags_out[tag_id] = tag
        print("{} Tags were exported.".format(len(tags_out)))
        return tags_out