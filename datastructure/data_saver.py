import os
import numpy as np

from utils.utils import check_n_make_dir


class DataSaver:
    def __init__(self, data_directory):
        self.data_dir = data_directory
        check_n_make_dir(data_directory)

        self.storage = dict()

    def add(self, i, data):
        fname = os.path.join(self.data_dir, "{}.npy".format(i))
        self.storage[i] = fname
        np.save(fname, data)

    def get(self, i):
        return np.load(self.storage[i])

    def get_special(self, i):
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        data = np.load(self.storage[i])
        np.load = np_load_old
        return data

    def load_storage(self):
        for i in os.listdir(self.data_dir):
            if i.endswith(".npy"):
                self.storage[i.replace(".npy", "")] = os.path.join(self.data_dir, i)

    def clear_storage(self):
        for i in self.storage:
            os.remove(self.storage[i])
