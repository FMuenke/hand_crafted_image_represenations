import numpy as np
import multiprocessing
from skimage import feature

from classic_image_classification.data_structure.image_handler import ImageHandler
from classic_image_classification.data_structure.matrix_handler import MatrixHandler


class GrayCoMatrix:
    def __init__(self):
        self.distances = [1, 2]
        self.angles = [0, np.pi / 2]
        self.levels = 64
        self.pool_count = multiprocessing.cpu_count()

    def _compute_for_roi(self, roi):
        g = feature.graycomatrix(roi,
                                 self.distances,
                                 self.angles,
                                 levels=self.levels,
                                 normed=True,
                                 symmetric=True)
        desc = [
            np.expand_dims(feature.graycoprops(g, 'contrast').ravel(), axis=0),
            np.expand_dims(feature.graycoprops(g, "dissimilarity").ravel(), axis=0),
            np.expand_dims(feature.graycoprops(g, "homogeneity").ravel(), axis=0),
            np.expand_dims(feature.graycoprops(g, "ASM").ravel(), axis=0),
            np.expand_dims(feature.graycoprops(g, "energy").ravel(), axis=0),
            np.expand_dims(feature.graycoprops(g, "correlation").ravel(), axis=0),
        ]
        return np.concatenate(desc, axis=1)

    def _build_regions_of_interest(self, image, key_points):
        regions_of_interest = []
        mat_h = MatrixHandler(image)
        mat_reduced = self.levels * mat_h.normalize()
        mat_h = MatrixHandler(mat_reduced.astype(np.int64))
        for x, y, roi_size in key_points:
            regions_of_interest.append(mat_h.cut_roi([x, y], roi_size))
        return regions_of_interest

    def compute(self, image, key_points):
        img_h = ImageHandler(image)
        regions_of_interest = self._build_regions_of_interest(img_h.gray(), key_points)

        with multiprocessing.Pool(self.pool_count) as p:
            descriptor_set = p.map(self._compute_for_roi, regions_of_interest)

        if len(descriptor_set) == 0:
            return None
        if len(descriptor_set) == 1:
            return descriptor_set[0]
        else:
            return np.concatenate(descriptor_set, axis=0)
