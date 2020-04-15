import copy


class BoundingBoxProcessor:
    """
    Class is used to handle all bounding box after processing (merging, conflict solving, ...)
    This class only works with bounding box format
    bboxes = [box, box, box, ...]
    box = [class_name, x1_abs, y1_abs, x2_abs, y2_abs, prob]
    use functions of class to format correctly
    """

    def __init__(
        self,
        merge_iou=0.1,
        conflict_iou=0.7,
        conflict_solving_strategy="probability",
        dominance_order=None,
        targeted_classes=None,
    ):
        self.merge_iou = merge_iou
        self.conflict_iou = conflict_iou

        self.conflict_solving_strategy = conflict_solving_strategy
        self.dominance_order = dominance_order

        self.targeted_classes = targeted_classes

        if dominance_order is not None:
            self.dominance_order = {}
            for order, cls in enumerate(dominance_order):
                self.dominance_order[cls] = order
        else:
            self.dominance_order = None

    def _get_class_names_from_bboxes(self, bboxes):
        class_names = []
        for box in bboxes:
            if box[0] not in class_names:
                class_names.append(box[0])
        return class_names

    def split_bboxes_from_prob(self, bboxes):
        bboxes_wo_prob = []
        prob = []
        for box in bboxes:
            bboxes_wo_prob.append([box[0], box[1], box[2], box[3], box[4]])
            prob.append(box[5])
        return bboxes_wo_prob, prob

    def add_prob_to_bboxes(self, bboxes_wo_prob, prob):
        bboxes = []
        for idx, box in enumerate(bboxes_wo_prob):
            bboxes.append([box[0], box[1], box[2], box[3], box[4], prob[idx]])
        return bboxes

    def prz_to_abs(self, bboxes, height, width):
        for box in bboxes:
            box[1] = box[1] * width
            box[2] = box[2] * height
            box[3] = box[3] * width
            box[4] = box[4] * height
        return bboxes

    def abs_to_prz(self, bboxes, height, width):
        for box in bboxes:
            box[1] = box[1] / width
            box[2] = box[2] / height
            box[3] = box[3] / width
            box[4] = box[4] / height
        return bboxes

    def merge_bboxes(self, bboxes):
        # Repeats itself until no merge was made
        repeat_flag = True
        approved_bboxes, approved_prob = self.split_bboxes_from_prob(bboxes)
        class_names = self._get_class_names_from_bboxes(bboxes)
        while repeat_flag:
            approved_bboxes, approved_prob, repeat_flag = self._merge_bboxes_iteration(
                approved_bboxes, approved_prob, class_names=class_names
            )
        bboxes = self.add_prob_to_bboxes(approved_bboxes, approved_prob)
        return bboxes

    def _merge_bboxes_iteration(self, bboxes, prob, class_names):
        approved_bboxes = []
        approved_prob = []
        repeat_flag = False
        for cls_id in class_names:
            for box_idx, bbox in enumerate(bboxes):
                if bbox[0] == cls_id:
                    box_is_approved = True
                    merge_target = -1
                    for ap_idx, ap_bbox in enumerate(approved_bboxes):
                        if ap_bbox[0] == cls_id:
                            ov = self._iou(ap_bbox[1:], bbox[1:])
                            if ov > self.merge_iou:  # Do BBoxes Overlap?
                                box_is_approved = False
                                merge_target = ap_idx
                            elif self._is_bbox_in_bbox(ap_bbox[1:], bbox[1:]):
                                box_is_approved = False
                                merge_target = ap_idx
                    if not box_is_approved:
                        # merge bboxes with each other
                        x1_merg = min(approved_bboxes[merge_target][1], bbox[1])
                        y1_merg = min(approved_bboxes[merge_target][2], bbox[2])
                        x2_merg = max(approved_bboxes[merge_target][3], bbox[3])
                        y2_merg = max(approved_bboxes[merge_target][4], bbox[4])

                        approved_bboxes[merge_target] = [
                            bbox[0],
                            x1_merg,
                            y1_merg,
                            x2_merg,
                            y2_merg,
                        ]
                        approved_prob[merge_target] = max(
                            prob[box_idx], approved_prob[merge_target]
                        )

                        repeat_flag = (
                            True
                        )  # merge was made, need to be merged again, until no merge happens

                    else:
                        # Add BBox to approved bboxes (no need for merging)
                        approved_bboxes.append(bbox)
                        approved_prob.append(prob[box_idx])

        return approved_bboxes, approved_prob, repeat_flag

    def detect_conflicting_bboxes(self, bboxes):
        conflict_pairs = []
        for idx1, box1 in enumerate(bboxes):
            for idx2, box2 in enumerate(bboxes):
                # do not test a pair twice or with itself
                if idx2 > idx1:
                    if self._iou(box1[1:-1], box2[1:-1]) > self.conflict_iou:
                        conflict_pairs.append([idx1, idx2])
        return conflict_pairs

    def _solve_with_probability(self, box1, box2):
        if box1[5] > box2[5]:
            dominant_box = box1
        else:
            dominant_box = box2
        return dominant_box

    def _solve_with_dominance(self, box1, box2):
        if self.dominance_order[box1[0]] < self.dominance_order[box2[0]]:
            dominant_box = box1
        else:
            dominant_box = box2
        return dominant_box

    def solve_conflicting_bboxes(self, bboxes):
        conflict_pairs = self.detect_conflicting_bboxes(bboxes)
        solved_bboxes = []
        solved_idxs = []
        for idx1, idx2 in conflict_pairs:
            box1 = bboxes[idx1]
            box2 = bboxes[idx2]
            if self.conflict_solving_strategy == "probability":
                dominant_box = self._solve_with_probability(box1, box2)
            elif self.conflict_solving_strategy == "dominance":
                assert (
                    self.dominance_order is not None
                ), "Error: Processor must be initialised with dominance order"
                if box1[0] in self.dominance_order and box2[0] in self.dominance_order:
                    dominant_box = self._solve_with_dominance(box1, box2)
                else:
                    dominant_box = self._solve_with_probability(box1, box2)
            else:
                raise ValueError(
                    "Unknown conflict solving strategy: {}".format(
                        self.conflict_solving_strategy
                    )
                )
            solved_idxs.append(idx1)
            solved_idxs.append(idx2)
            solved_bboxes.append(dominant_box)
        # Add unconflicting bboxes
        for idx in range(len(bboxes)):
            if idx not in solved_idxs:
                solved_bboxes.append(bboxes[idx])
                solved_idxs.append(idx)
        return solved_bboxes

    def targeted_solve_conflicting_bboxes(self, bboxes):
        assert (
            self.targeted_classes is not None
        ), "Error: Processor must be initialised with targeted_classes"
        # Add not confilcting classes to target classes otherwise they would be lost
        class_names = self._get_class_names_from_bboxes(bboxes)
        for class_name in class_names:
            class_name_is_included = False
            for class_target_group in self.targeted_classes:
                if class_name in class_target_group:
                    class_name_is_included = True
            if not class_name_is_included:
                self.targeted_classes.append(class_name)
        # Bounding-Boxes with defined classes
        bboxes_out = None
        for class_target_group in self.targeted_classes:
            bboxes_copy = copy.deepcopy(bboxes)
            bboxes_copy = self.filter_bboxes_by_classes(bboxes_copy, class_target_group)
            bboxes_copy = self.solve_conflicting_bboxes(bboxes_copy)
            if bboxes_out is None:
                bboxes_out = bboxes_copy
            else:
                bboxes_out.extend(bboxes_copy)
        return bboxes_out

    def filter_bboxes_by_classes(self, bboxes, class_names):
        """
        Used to delete unwanted classes in bounding_boxes.
        Args:
            bboxes: list of bounding boxes
            class_names: list of class_name strings you want to KEEP

        Returns:
            bboxes_filtered: List of bounding boxes with valid classes

        """
        if type(class_names) is not list:
            class_names = [class_names]
        bboxes_filtered = []
        for box in bboxes:
            if box[0] in class_names:
                bboxes_filtered.append(box)
        return bboxes_filtered

    def filter_bboxes_by_confidence(self, bboxes, conf_thresh):
        bboxes_filtered = []
        for box in bboxes:
            if box[5] > conf_thresh:
                bboxes_filtered.append(box)
        return bboxes_filtered

    def _union(self, au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    def _intersection(self, ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w * h

    def iou(self, box1, box2):
        a = box1[1:4]
        b = box2[1:4]
        return self._iou(a, b)

    def _iou(self, a, b):
        # a and b should be (x1,y1,x2,y2)

        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0

        area_i = self._intersection(a, b)
        area_u = self._union(a, b, area_i)

        return float(area_i) / float(area_u + 1e-6)

    def _is_bbox_in_bbox(self, a, b):
        area_i = self._intersection(a, b)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_min = min(area_a, area_b)
        if area_i / (area_min + 1e-6) > 0.5:
            return True
        else:
            return False
