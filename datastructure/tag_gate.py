import numpy as np


class TagGate:
    def __init__(self, requirements, down_sample=0):
        self.req = requirements
        self.down_sample = down_sample

    def apply(self, tag_set):
        to_delete = []
        print("Checking tag requirements...")
        for tid in tag_set:
            x1 = tag_set[tid].box[1]
            y1 = tag_set[tid].box[2]
            x2 = tag_set[tid].box[3]
            y2 = tag_set[tid].box[4]

            if "height" in self.req:
                if (y2 - y1) < self.req["height"]:
                    to_delete.append(tid)

            if "width" in self.req:
                if (x2 - x1) < self.req["width"]:
                    to_delete.append(tid)

            if np.random.randint(0, 100) < 100 * self.down_sample:
                to_delete.append(tid)

        for tid in to_delete:
            if tid in tag_set:
                del tag_set[tid]

        print("Blocking {} Tags from further processing.".format(len(to_delete)))

        return tag_set
