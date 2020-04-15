import cv2


class KeyPointSet:
    def __init__(self, key_point_mode, sampling_steps=None, sampling_window=None, mask=None):
        self.key_point_mode = key_point_mode
        self.sampling_steps = sampling_steps
        self.sampling_window = sampling_window
        self.mask = mask

    def mask_key_points(self, key_points):
        key_points_masked = []
        for cx, cy, s in key_points:
            if self.mask[cy, cx] == 1:
                key_points_masked.append([cx, cy, s])
        return key_points_masked

    def mask_open_cv_key_points(self, open_cv_key_points):
        key_points_masked = []
        for kp_cv in open_cv_key_points:
            if self.mask[kp_cv.pt[1], kp_cv.pt[0]] == 1:
                key_points_masked.append(kp_cv)
        return key_points_masked

    def _build_open_cv_key_point_detectors(self):
        open_cv_key_point_detectors = {
            "orb": cv2.ORB_create(),
            "akaze": cv2.AKAZE_create(),
            "kaze": cv2.KAZE_create(),
            "brisk": cv2.BRISK_create(),
        }
        return open_cv_key_point_detectors

    def _define_dense_key_point_grid(self, image):
        height, width = image.shape[:2]
        step = self.sampling_steps
        key_points = []
        for j in range(step, height - step, step):
            for i in range(step, width - step, step):
                key_points.append([i, j, self.sampling_window])
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, (width, height), cv2.INTER_NEAREST)
            key_points = self.mask_key_points(key_points)
        if len(key_points) == 0:
            key_points.append([width/2, height/2, self.sampling_window])
        return key_points

    def _get_one_key_point(self, image):
        height, width = image.shape[:2]
        cx = int(width / 2)
        cy = int(height / 2)
        s = max(height, width)
        return [[cx, cy, s]]

    def _detect_open_cv_key_points(self, image):
        open_cv_extractors = self._build_open_cv_key_point_detectors()
        kp_detector = open_cv_extractors[self.key_point_mode]
        open_cv_key_points = kp_detector.detect(image)
        if self.mask is not None:
            height, width = image.shape[:2]
            self.mask = cv2.resize(self.mask, (width, height), cv2.INTER_NEAREST)
            open_cv_key_points = self.mask_open_cv_key_points(open_cv_key_points)
        if len(open_cv_key_points) == 0:
            height, width = image.shape[:2]
            open_cv_key_points.append(cv2.KeyPoint(int(width/2), int(height/2), self.sampling_window, _class_id=0))
        return open_cv_key_points

    def _key_points_to_open_cv_key_points(self, key_points):
        open_cv_key_points = []
        for x, y, s in key_points:
            open_cv_key_points.append(cv2.KeyPoint(x, y, s, _class_id=0))
        return open_cv_key_points

    def _open_cv_key_points_to_key_points(self, open_cv_key_points):
        key_points = []
        for open_cv_kp in open_cv_key_points:
            key_points.append([open_cv_kp.pt[0], open_cv_kp.pt[1], open_cv_kp.size])
        return key_points

    def get_key_points(self, image):
        if self.key_point_mode == "dense":
            return self._define_dense_key_point_grid(image)
        elif self.key_point_mode == "one":
            return self._get_one_key_point(image)
        else:
            open_cv_key_points = self._detect_open_cv_key_points(image)
            return self._open_cv_key_points_to_key_points(open_cv_key_points)

    def get_open_cv_key_points(self, image):
        if self.key_point_mode == "dense":
            key_points = self._define_dense_key_point_grid(image)
            open_cv_key_points = self._key_points_to_open_cv_key_points(key_points)
        else:
            open_cv_key_points = self._detect_open_cv_key_points(image)
        return open_cv_key_points

