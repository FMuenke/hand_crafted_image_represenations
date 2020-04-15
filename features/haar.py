import cv2
import numpy as np
import multiprocessing

from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image


def basic_haar_features(roi):
    try:
        roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(e)
        print(roi.shape)
        roi = np.zeros((20, 20))
    roi = integral_image(roi)
    f1 = np.sum(roi[10:, :] - roi[:10, :]) / (20 * 20)
    f2 = np.sum(roi[:, 10:] - roi[:, :10]) / (20 * 20)
    f3 = np.sum(roi[10:, 10:] - roi[:10, :10]) / (20 * 20)
    f4 = np.sum(roi[10:, :10] - roi[:10, 10:]) / (20 * 20)
    return [f1, f2, f3, f4]


if __name__ == "__main__":
    img = np.ones((20, 20), dtype=np.uint8)
    img_ii = integral_image(img)
    fh = basic_haar_features(img_ii)
    print(fh)
    print(len(fh))

from datastructure.image_handler import ImageHandler
from datastructure.matrix_handler import MatrixHandler


class Haar:
    def __init__(self):
        self.pool_count = multiprocessing.cpu_count()

    def _compute_for_roi(self, region_of_interest):
        height, width = region_of_interest.shape[:2]
        desc = haar_like_feature(region_of_interest, width=width, height=height, r=0, c=0)
        desc = np.reshape(desc, (1, -1))

        return desc

    def _build_regions_of_interest(self, image, key_points):
        regions_of_interest = []
        mat_h = MatrixHandler(image)
        for x, y, roi_size in key_points:
            regions_of_interest.append(mat_h.cut_roi([x, y], roi_size))
        return regions_of_interest

    def compute(self, image, key_points):
        img_h = ImageHandler(image)
        regions_of_interest = self._build_regions_of_interest(img_h.integral(), key_points)

        with multiprocessing.Pool(self.pool_count) as p:
            descriptor_set = p.map(self._compute_for_roi, regions_of_interest)

        if len(descriptor_set) == 0:
            return None
        if len(descriptor_set) == 1:
            return descriptor_set[0]
        else:
            return np.concatenate(descriptor_set, axis=0)
