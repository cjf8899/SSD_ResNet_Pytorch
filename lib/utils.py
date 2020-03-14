

import numpy as np
import torch
import cv2




def point_form(boxes):
    

    tl = boxes[:, :2] - boxes[:, 2:]/2
    br = boxes[:, :2] + boxes[:, 2:]/2

    return np.concatenate([tl, br], axis=1)


def detection_collate(batch):
    
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs), np.array(targets)



def bbox_iou(box_a, box_b):
   
    m = box_a.shape[0]
    n = box_b.shape[0]

    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.maximum(br-tl, 0)
    
    inner = wh[:, :, 0]*wh[:, :, 1]

    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]

    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]

    a = a[:, None]
    b = b[None, :]

    

    return inner / (a+b-inner)


def nms(boxes, score, threshold=0.4):
   

    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break

        sort_ids = sort_ids[:-1]
        box = boxes[i].reshape(1, 4)
        ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)

        sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])

    return pick




def detect(locations, scores, nms_threshold, gt_threshold):
   

    scores = scores[:, 1:] 

    keep_boxes = []
    keep_confs = []
    keep_labels = []
    
    for i in range(scores.shape[1]):
        mask = scores[:, i] >= gt_threshold
        label_scores = scores[mask, i] 
        label_boxes = locations[mask]
        if len(label_scores) == 0:
            continue

        pick = nms(label_boxes, label_scores, threshold=nms_threshold)
        label_scores = label_scores[pick]
        label_boxes = label_boxes[pick]
        

        keep_boxes.append(label_boxes.reshape(-1))
        keep_confs.append(label_scores)
        keep_labels.extend([i]*len(label_scores))
    
    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    
    keep_boxes = np.concatenate(keep_boxes, axis=0).reshape(-1, 4)

    keep_confs = np.concatenate(keep_confs, axis=0)
    keep_labels = np.array(keep_labels).reshape(-1)


    return keep_boxes, keep_confs, keep_labels





def draw_rectangle(src_img, labels, conf, locations, label_map):
    
    num_obj = len(labels)
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    img = src_img.copy()
    for i in range(num_obj):
        tl = tuple(locations[i][:2])
        br = tuple(locations[i][2:])
        
        cv2.rectangle(img,
                      tl,
                      br,
                      COLORS[i%3], 3)
        cv2.putText(img, label_map[labels[i]], tl,
                    FONT, 1, (255, 255, 255), 2)
    
    img = img[:, :, ::-1]

    return img