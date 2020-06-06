import numpy as np
import random
from sklearn.model_selection import train_test_split


class DataSampler:
    def __init__(self, class_mapping, x=None, y=None, mode="evenly"):
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

        if self.x is not None and self.y is not None:
            self._compile_x_y()

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
        if self.mode == "random":
            return self._random_x_y(percentage)

        if self.mode == "fixed":
            return self._fixed_x_y(percentage)

        raise ValueError("Mode {} not recognised".format(self.mode))

    def _fixed_x_y(self, percentage):
        b = int((1 - percentage) * self.x.shape[0])
        x_train = self.x[:b, :]
        y_train = self.y[:b]
        x_test = self.x[b:, :]
        y_test = self.y[b:]
        return x_train, x_test, y_train, y_test

    def _random_x_y(self, percentage):
        return train_test_split(self.x, self.y, test_size=percentage)
