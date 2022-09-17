import cv2
from PIL import Image
import os
import numpy as np

from classic_image_classification.utils.utils import check_n_make_dir


class BoxTag:
    tag_type = "box"

    def __init__(self,
                 tag_id,
                 path_to_image,
                 tag_class,
                 box,
                 class_mapping):
        self.tag_id = tag_id
        self.image_id = path_to_image
        self.box = box
        self.tag_class = tag_class

        if class_mapping is None:
            class_mapping = {}

        self.class_mapping = class_mapping
        self.class_mapping_inv = {v: k for k, v in self.class_mapping.items()}

        self.compile_classes()

    def __str__(self):
        return "Tag: {}\n - Img: {}\n - Box: {}".format(self.tag_id, self.image_id, self.box)

    def is_valid(self):
        if not os.path.isfile(self.image_id):
            print("Tag not valid: {}".format(self.image_id))
            return False

        if self.box[1] < self.box[3] and self.box[2] < self.box[4]:
            pass
        else:
            print("Tag not valid: {}".format(self.image_id))
            return False

        data = self.load_data()
        height, width = data.shape[:2]
        if height < 1 or width < 1:
            print("Tag not valid: {}".format(self.image_id))
            return False

        return True

    def compile_classes(self):
        if len(self.class_mapping) != 0:
            filtered_classes = []
            for cls in self.tag_class:
                if cls in self.class_mapping:
                    filtered_classes.append(cls)

            if len(filtered_classes) == 0:
                filtered_classes = ["bg"]
            self.tag_class = filtered_classes

    def load_data(self, surrounding=0):
        assert os.path.isfile(self.image_id), "Error: {} not found.".format(self.image_id)
        img = cv2.imread(self.image_id)
        img_height, img_width = img.shape[:2]
        box_width = self.box[3] - self.box[1]
        box_height = self.box[4] - self.box[2]
        x1 = max(0, int(self.box[1] - surrounding * box_width))
        y1 = max(0, int(self.box[2] - surrounding * box_height))
        x2 = min(img_width, int(self.box[3] + surrounding * box_width))
        y2 = min(img_height, int(self.box[4] + surrounding * box_height))
        data = img[y1:y2, x1:x2, :]
        return data

    def load_y(self):
        assert len(self.tag_class) == 1, "Multi Classification not supported {}".format(self.tag_class)
        y_tag = -1
        if self.tag_class[0] in self.class_mapping:
            y_tag = self.class_mapping[self.tag_class[0]]
        return np.array(y_tag)

    def has_relevant_classes(self, classes_to_consider):
        for cls in classes_to_consider:
            if cls in self.tag_class:
                return True
        return False

    def export(self, export_folder, surrounding=0):
        check_n_make_dir(export_folder)
        class_img_dir = os.path.join(export_folder, "images")
        class_lbf_dir = os.path.join(export_folder, "labels")
        self.export_box(class_img_dir, surrounding=surrounding)
        self._export_label(class_lbf_dir)

    def generate_file_name(self, export_folder):
        file_name = "tag"
        for cls in self.tag_class:
            file_name += "_{}".format(cls)
        file_name += "_{}".format(os.path.basename(self.image_id))
        return os.path.join(export_folder, file_name)

    def export_box(self, export_folder, surrounding=0, with_text=False):
        check_n_make_dir(export_folder)
        box_data = self.load_data(surrounding=surrounding)
        box_name = "{}.jpg".format(self.generate_file_name(export_folder))
        if with_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(box_data, "{}".format(self.tag_class), (10, 10), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(box_name, box_data)

    def _export_label(self, export_folder):
        check_n_make_dir(export_folder)
        box_name = "{}.txt".format(self.generate_file_name(export_folder))
        lb_str = ""
        for cls in self.tag_class:
            lb_str += cls
            lb_str += " "
        lb_str = lb_str[:-1]
        with open(box_name, "w") as f:
            f.write("{}".format(lb_str))

    def write_prediction(self, prediction, prediction_folder):
        check_n_make_dir(prediction_folder)
        p_string = ""
        for p in prediction:
            if p in self.class_mapping_inv:
                p_string += str(self.class_mapping_inv[p])
            else:
                p_string += p
            p_string += "_"
        p_string = p_string[:-1]

        cls_folder = os.path.join(prediction_folder, p_string)
        check_n_make_dir(cls_folder)
        self.export_box(cls_folder, with_text=True)

    def evaluate_prediction(self, prediction, result_dict):
        prediction_string = []
        for i, p in enumerate(prediction):
            if p in self.class_mapping_inv:
                prediction_string.append(self.class_mapping_inv[p])
            else:
                prediction_string.append(p)
        for cls in result_dict:
            if cls in self.tag_class:
                if cls in prediction_string:
                    result_dict[cls]["tp"] += 1
                    result_dict["overall"]["tp"] += 1
                else:
                    result_dict[cls]["fn"] += 1
                    result_dict["overall"]["fn"] += 1
            else:
                if cls in prediction_string:
                    result_dict[cls]["fp"] += 1
                    result_dict["overall"]["fp"] += 1
        return result_dict

    def get_data_size(self):
        im = Image.open(self.image_id)
        width, height = im.size
        return height, width

    def remove(self):
        os.remove(self.image_id)
        label_id = self.image_id.replace("images", "labels")
        label_id = label_id.replace(".jpg", ".txt")
        os.remove(label_id)
