import numpy
from ultralytics import YOLO

class Tracker():

    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def update(self, image):
        bbox = []

        # detect and track only the people (class 0 in COCO dataset)
        results = self.model.track(source=image, persist=True, tracker="botsort.yaml",classes=0)

        for r in results:
            boxes = r.boxes
            for i,box in enumerate(boxes):
                sublist = [] 
                id = box.id[i]
                # get box coordinates in (left, top, right, bottom) format
                # convert the box type to something "digestible", not floats 
                b = box.xyxy[i].numpy().astype(numpy.int32)  
                sublist.append(b)
                sublist.append(id)
                bbox.append(sublist)

        return image,bbox

