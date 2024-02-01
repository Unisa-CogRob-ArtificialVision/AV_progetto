import numpy
from ultralytics import YOLO

class Tracker():

    def __init__(self, **kwargs):
        self.model = YOLO('DETECTOR_models/yolov8x.pt')

    def update(self, image):
        bbox = []

        # detect and track only the people (class 0 in COCO dataset)
        results = self.model.track(source=image, persist=True, tracker="botsort.yaml",classes=0)
        try:
            boxes = results[0].boxes.xyxy
            ids = results[0].boxes.id.int().cpu().tolist()
            for i,box in enumerate(boxes):
                sublist = [] 
                id = ids[i]
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())  
                bbox.append([x1,y1,x2,y2,id])

        except Exception:
            bbox = []
        return image,bbox

