
    
import itertools
import numpy as np

from lib.utils import bbox_iou
from lib.utils import point_form

class MultiBoxEncoder(object):

    

    def __init__(self, opt):
        self.variance = opt.variance
        default_boxes = list()
        
        for k in range(len(opt.grids)):
            for v, u in itertools.product(range(opt.grids[k]), repeat=2):
                cx = (u + 0.5) * opt.steps[k]
                cy = (v + 0.5) * opt.steps[k]

                s = opt.sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(opt.sizes[k] * opt.sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = opt.sizes[k]
                for ar in opt.aspect_ratios[k]:
                    default_boxes.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_boxes.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))

        default_boxes = np.clip(default_boxes, a_min=0, a_max=1)
        self.default_boxes = np.array(default_boxes)

    def encode(self, boxes, labels, threshold=0.5):
       
        if len(boxes) == 0:
            return (
                np.zeros(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        iou = bbox_iou(point_form(self.default_boxes), boxes)


        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        boxes = boxes[gt_idx]
        labels = labels[gt_idx]

        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + labels
        conf[iou < threshold] = 0
       

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc):
        
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes


if __name__ == '__main__':
    from config import opt
    mb = MultiBoxEncoder(opt)

    print(mb.default_boxes[:10])


