import sys
#sys.path.insert(0, './YOLOX')

from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from utils.visualize import vis_track
from strong_sort import StrongSORT


class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='yolox_s.pth', gpu=True):
        self.detector =  YOLO('DETECTOR_models/yolov8l.pt') # load yolov8 detector from ultralytics
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        print('CONFIG DEEPSORT: ', cfg.DEEPSORT)
        self.strongsort = StrongSORT(model_weights='osnet_ain_x1_0_msmt17.pt',
                            device='cuda',max_dist=cfg.DEEPSORT.MAX_DIST,
                            #min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            #nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, 
                            n_init=cfg.DEEPSORT.N_INIT, 
                            nn_budget=cfg.DEEPSORT.NN_BUDGET)
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
        results = self.detector(source=image,classes=0, show_labels=False,show_conf=False, show_boxes=False)
        try:
            boxes = results[0].boxes.xywh
            for i,box in enumerate(boxes):
                x1, y1, w, h = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                #print(h)
                if h < 100:
                    continue
                scores.append(results[0].boxes.conf[i])
                bbox.append([x1,y1,w,h])
            bbox_xywh = torch.Tensor(bbox)
            #outputs = self.strongsort.update(bbox_xywh, torch.tensor(scores), torch.tensor(0), ori_img=image)    # update tracker state
            outputs = self.deepsort.update(bbox_xywh, scores, image)    # update tracker state
            
            #image = vis_track(image, outputs)

        except Exception:   # empty results
            bbox = []
            outputs = []

        return image, outputs

