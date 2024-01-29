import sys
sys.path.insert(0, './YOLOX')

from ultralytics import YOLO

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from utils.visualize import vis_track


class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='yolox_s.pth', gpu=False):
        self.detector =  YOLO('DETECTOR_models/yolov8s.pt')
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, 
                            n_init=cfg.DEEPSORT.N_INIT, 
                            nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=gpu)
        self.filter_class = filter_class

    def update(self, image):
        bbox = []
        scores = []
        # detect and track only the people (class 0 in COCO dataset)
        with HiddenPrints():
            results = self.detector(source=image,classes=0, show_labels=False,show_conf=False, show_boxes=False)
        try:
            boxes = results[0].boxes.xywh
            for i,box in enumerate(boxes):
                scores.append(results[0].boxes.conf[i])
                x1, y1, w, h = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())  
                bbox.append([x1,y1,w,h])
            bbox_xywh = torch.Tensor(bbox)
            outputs = self.deepsort.update(bbox_xywh, scores, image)
            image = vis_track(image, outputs)

        except Exception:
            bbox = []
            outputs = []
        return image, outputs

import os
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout