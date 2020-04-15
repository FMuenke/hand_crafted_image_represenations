import matplotlib.pyplot as plt
from umap import UMAP
import cv2
import numpy as np
import os


class Visualizer:
    def __init__(self):
        self.x = None
        self.y = None

        self._reducer = None
        self._reducer_super = None
        self._reducer_target = None

    def set_xy(self, x, y):
        self.x = x
        self.y = y

    def fit_reducer(self):
        if self.x is not None:
            if not type(self.x) is np.ndarray:
                self.x = np.concatenate(self.x, axis=0)
                self.y = np.array(self.y)

            self._reducer = UMAP()
            self._reducer_super = UMAP()

            self._reducer.fit(self.x)
            self._reducer_super.fit(self.x, self.y)

    def _scatter(self, x1, x2, y, title, save_path=None):
        fig, ax = plt.subplots()
        scatter = plt.scatter(x1, x2, c=y, alpha=0.35)
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        ax.add_artist(legend1)
        fig.suptitle('Scatter_{}'.format(title), fontsize=14)
        if save_path is None:
            plt.show()
        else:
            dir_name = os.path.dirname(save_path)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print("Plot saved to {}".format(save_path))

    def scatter(self, i1, i2, title, save_path=None):
        self._scatter(self.x[:, i1], self.x[:, i2], self.y, title, save_path)

    def scatter_reduced(self, title, save_path=None, supervised=False):
        if supervised:
            x = self._reducer_super.transform(self.x)
        else:
            x = self._reducer.transform(self.x)
        self._scatter(x[:, 0], x[:, 1], self.y, title=title, save_path=save_path)

    def plot_image(self, path_to_image, box_sets, result_path):
        def markBBoxesOnImg(img, bboxes, mark_color, prb_plot=False):
            # Markiert alle BBoxes samt Labels auf dem Bild (ABSOLUTE)
            for index, item in enumerate(bboxes):
                # BoundingBox
                bb_x1 = int(bboxes[index][1])
                bb_y1 = int(bboxes[index][2])
                bb_x2 = int(bboxes[index][3])
                bb_y2 = int(bboxes[index][4])
                cv2.rectangle(img, (bb_x1, bb_y1), (bb_x2, bb_y2), mark_color, 2)

                classname = bboxes[index][0]
                if prb_plot:
                    probability = bboxes[index][5]
                    text_label = "{} {}".format(
                        classname, round(100 * float(probability)) / 100
                    )
                else:
                    text_label = "{}".format(classname)
                (ret_val, base_line) = cv2.getTextSize(
                    text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1
                )
                if prb_plot:
                    text_org = (bb_x1, bb_y1 + base_line)
                else:
                    text_org = (bb_x1, bb_y2 - base_line)

                cv2.putText(
                    img,
                    text_label,
                    text_org,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    mark_color,
                    1,
                )
            return np.copy(img)

        img = cv2.imread(path_to_image)
        for box_set in box_sets:
            img = markBBoxesOnImg(img, box_set[0], box_set[1])
        cv2.imwrite(result_path, img)

