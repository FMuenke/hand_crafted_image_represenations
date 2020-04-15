import numpy as np
import random
from sklearn.model_selection import train_test_split


class DataSampler:
    def __init__(self, class_mapping, tag_set=None, x=None, y=None, mode="evenly"):
        self.tags = tag_set
        self.x = x
        self.y = y
        self.mode = mode
        self.clmp = class_mapping
        self.clnb = None
        self.clwt = None

        self.compile()

    def compile(self):
        """
        While images are loaded all classes are counted. With this function
        the class weights for training can be adjusted.
        If specified in config (cfg.use_class_weights).
        Returns:
            clwt (list): list with class weights sorted like class mapping clmp
        """
        if self.tags is not None:
            self._compile_tags()

        if self.x is not None and self.y is not None:
            self._compile_x_y()

    def _compile_tags(self):
        class_numbers = {}
        print("Analyzing Tags...")
        for tag_id in self.tags:
            tag = self.tags[tag_id]
            for cls in tag.tag_class:
                if cls in class_numbers:
                    class_numbers[cls] += 1
                else:
                    class_numbers[cls] = 1
        print("Sample Distribution")
        for cls in self.clmp:
            print("> {}, {} : {}".format(self.clmp[cls], cls, class_numbers[cls]))

        clwt = [0] * len(self.clmp)
        clnb = [0] * len(self.clmp)
        for idx, cls in enumerate(class_numbers):
            clnb[idx] = class_numbers[cls]
            clwt[idx] = 1 - class_numbers[cls] / len(self.tags)

        self.clwt = clwt
        self.clnb = clnb

        tag_set = []
        for tag_id in self.tags:
            tag_set.append(self.tags[tag_id])
        self.tags = np.array(tag_set)

    def _compile_x_y(self):
        class_numbers = {}
        print("Analyzing Tags...")
        counter = 0
        for y_img in self.y:
            tag_class = y_img
            counter += 1
            if tag_class in class_numbers:
                class_numbers[tag_class] += 1
            else:
                class_numbers[tag_class] = 1
        print("Sample Distribution")
        for cls in class_numbers:
            print("> {} : {}".format(cls, class_numbers[cls]))

        clwt = [0] * len(self.clmp)
        clnb = [0] * len(self.clmp)
        for idx, cls in enumerate(class_numbers):
            clnb[idx] = class_numbers[cls]
            clwt[idx] = class_numbers[cls] / counter

        self.clwt = clwt
        self.clnb = clnb

    def train_test_split(self, percentage=0.2):
        if self.mode == "evenly":
            return self._deterministic_evenly_distributed(percentage)
        if self.mode == "random":
            return self._random(percentage)

        raise ValueError("Mode {} not recognised".format(self.mode))

    def _random(self, percentage):
        if self.x is not None:
            return self._random_x_y(percentage)
        if self.tags is not None:
            return self._random_tags(percentage)
        raise ValueError("No data available")

    def _random_tags(self, percentage):
        tags_train = []
        tags_test = []
        for t in self.tags:
            if random.randint(0, 100) > 100 * percentage:
                tags_train.append(t)
            else:
                tags_test.append(t)
        return tags_train, tags_test

    def _random_x_y(self, percentage):
        return train_test_split(self.x, self.y, test_size=percentage)

    def _deterministic_evenly_distributed(self, percentage):
        if self.tags is not None:
            return self._deterministic_evenly_distributed_tags(percentage)
        if self.x is not None:
            return self._deterministic_evenly_distributed_x_y(percentage)
        raise ValueError("No data available")

    def _deterministic_evenly_distributed_x_y(self, percentage):
        target_number = int(min(self.clnb) * percentage)

        nb_test = [0] * len(self.clnb)

        missing_samples = [target_number] * len(self.clnb)

        test_idxs = []
        train_idxs = []

        for idx, y_img in enumerate(self.y):
            y_one_hot = np.zeros(len(self.clmp))
            y_one_hot[y_img] = 1
            images_are_needed = missing_samples - y_one_hot > 0
            cls_on_image = y_one_hot > 0
            is_test = np.logical_and(images_are_needed, cls_on_image)
            if True in is_test:
                missing_samples -= y_one_hot
                nb_test += y_one_hot
                test_idxs.append(idx)
            else:
                train_idxs.append(idx)

        x_train = self.x[train_idxs]
        y_train = self.y[train_idxs]

        x_test = self.x[test_idxs]
        y_test = self.y[test_idxs]
        print("Testing on: {}".format(nb_test))
        return x_train, x_test, y_train, y_test

    def _deterministic_evenly_distributed_tags(self, percentage):
        """
        Function is used to split data in training und test dataset. With an even distribution of class samples in the test dataset
        Args:
            percentage: distribution of the datasplit
        """
        target_number = int(min(self.clnb) * percentage)

        nb_test = [0] * len(self.clnb)

        missing_samples = [target_number] * len(self.clnb)

        test_idxs = []
        train_idxs = []

        for idx, tag in enumerate(self.tags):
            y_img = tag.load_y(one_hot_encoding=True)
            images_are_needed = missing_samples - y_img > 0
            cls_on_image = y_img > 0
            is_test = np.logical_and(images_are_needed, cls_on_image)
            if True in is_test:
                missing_samples -= y_img
                nb_test += y_img
                test_idxs.append(idx)
            else:
                train_idxs.append(idx)

        tag_set_train = self.tags[train_idxs]
        tag_set_test = self.tags[test_idxs]
        print("Testing on: {}".format(nb_test))
        return tag_set_train, tag_set_test