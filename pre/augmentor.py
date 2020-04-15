import cv2
import numpy as np
'''
In this file are all functions for image augmentation stored
Important use following format for bboxes variabel!
bboxes = [bbox, bbox2, bbox3, ... ]
with bbox = [classname, x1_abs, y1_abs, x2_abs, y2_abs, (prob)]
'''


def apply_auto_contrast(img):
    for ch in range(img.shape[2]):
        equ = cv2.equalizeHist(img[:, :, ch])
        img[:, :, ch] = equ
    return img


def apply_noise(img, low=1, high=10):
    noise = np.random.randint(low=low, high=high, size=img.shape, dtype='uint8')
    img += noise
    return img


def apply_blurr(img):
    img = cv2.blur(img, (5, 5))
    return img


def apply_brightening(img):
    scale = np.random.random() + 0.5
    img = np.round(scale * img)
    return img


def apply_horizontal_flip(img):
    """
    Applys a horizontal flip to image and its annotated bounding-boxes
    :param img: original image
    :param bboxes: annotated bounding-boxes
    :return:
        img: flipped image
        bboxes: flipped bboxes
    """
    img = cv2.flip(img, 1)
    return img


def apply_vertical_flip(img):
    img = cv2.flip(img, 0)
    return img


def apply_crop(img):
    height, width, ch = img.shape
    prz_zoom = 0.10
    w_random = int(width * prz_zoom)
    h_random = int(height * prz_zoom)
    if w_random > 0:
        x1_img = np.random.randint(w_random)
        x2_img = width - np.random.randint(w_random)
    else:
        x1_img = 0
        x2_img = width

    if h_random > 0:
        y1_img = np.random.randint(h_random)
        y2_img = height - np.random.randint(h_random)
    else:
        y1_img = 0
        y2_img = height

    img = img[y1_img:y2_img, x1_img:x2_img, :]
    return img


def apply_rotation_90(img):
    angle = np.random.choice([0, 90, 180, 270])

    if angle == 270:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 0)
    elif angle == 180:
        img = cv2.flip(img, -1)
    elif angle == 90:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 1)
    elif angle == 0:
        pass

    return img


def apply_tiny_rotation(img):
    rows, cols = img.shape[:2]
    angle = np.random.randint(20) - 10
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return img


class Augmentor:
    def __init__(self):
        self.opt = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "noise": True,
            "auto_contrast": True,
            "brightening": True,
            "blur": True,
            "crop": True,
            "rotation": True,
        }

    def apply(self, img):

        # Augmentation (randomized)

        if 0 == np.random.randint(6) and self.opt["horizontal_flip"]:
            img = apply_horizontal_flip(img)

        if 0 == np.random.randint(6) and self.opt["vertical_flip"]:
            img = apply_vertical_flip(img)

        if 0 == np.random.randint(6) and self.opt["noise"]:
            img = apply_noise(img)

        if 0 == np.random.randint(10) and self.opt["auto_contrast"]:
            try:
                img = apply_auto_contrast(img)
            except:
                pass

        if 0 == np.random.randint(6) and self.opt["brightening"]:
            img = apply_brightening(img)

        if 0 == np.random.randint(6) and self.opt["blur"]:
            img = apply_blurr(img)

        if 0 == np.random.randint(6) and self.opt["crop"]:
            img = apply_crop(img)

        # if np.random.randint(4):
        #    img = apply_tiny_rotation(img)

        if 0 == np.random.randint(6) and self.opt["rotation"]:
            img = apply_rotation_90(img)

        return img
