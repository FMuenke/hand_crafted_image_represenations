import cv2


def key_points_to_open_cv_key_points(key_points):
    open_cv_key_points = []
    for x, y, s in key_points:
        open_cv_key_points.append(cv2.KeyPoint(x, y, s))
    return open_cv_key_points


def open_cv_key_points_to_key_points(open_cv_key_points):
    key_points = []
    for open_cv_kp in open_cv_key_points:
        key_points.append([open_cv_kp.pt[0], open_cv_kp.pt[1], open_cv_kp.size])
    return key_points


def build_open_cv_key_point_detectors():
    open_cv_key_point_detectors = {
        "orb": cv2.ORB_create(),
        "sift": cv2.SIFT_create(),
        "akaze": cv2.AKAZE_create(),
        "kaze": cv2.KAZE_create(),
        "brisk": cv2.BRISK_create(),
    }
    return open_cv_key_point_detectors


def get_one_key_point(image):
    height, width = image.shape[:2]
    cx = int(width / 2)
    cy = int(height / 2)
    s = max(height, width)
    return [[cx, cy, s]]


class KeyPointSet:
    def __init__(self, key_point_mode, sampling_steps=None, sampling_window=None):
        self.key_point_mode = key_point_mode
        self.sampling_steps = sampling_steps
        self.sampling_window = sampling_window

    def _define_dense_key_point_grid(self, image):
        height, width = image.shape[:2]
        step = self.sampling_steps
        key_points = []
        for j in range(step, height - step, step):
            for i in range(step, width - step, step):
                key_points.append([i, j, self.sampling_window])
        if len(key_points) == 0:
            key_points.append([width/2, height/2, self.sampling_window])
        return key_points

    def _detect_open_cv_key_points(self, image):
        open_cv_extractors = build_open_cv_key_point_detectors()
        kp_detector = open_cv_extractors[self.key_point_mode]
        open_cv_key_points = kp_detector.detect(image)
        return open_cv_key_points

    def get_key_points(self, image):
        if self.key_point_mode == "dense":
            return self._define_dense_key_point_grid(image)
        elif self.key_point_mode == "one":
            return get_one_key_point(image)
        else:
            open_cv_key_points = self._detect_open_cv_key_points(image)
            return open_cv_key_points_to_key_points(open_cv_key_points)

    def get_open_cv_key_points(self, image):
        if self.key_point_mode == "dense":
            key_points = self._define_dense_key_point_grid(image)
            open_cv_key_points = key_points_to_open_cv_key_points(key_points)
        elif self.key_point_mode == "one":
            key_points = get_one_key_point(image)
            open_cv_key_points = key_points_to_open_cv_key_points(key_points)
        else:
            open_cv_key_points = self._detect_open_cv_key_points(image)
        return open_cv_key_points
