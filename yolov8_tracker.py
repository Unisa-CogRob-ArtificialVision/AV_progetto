import numpy
from ultralytics import YOLO

class Tracker():

    def __init__(self, **kwargs):
        self.model = YOLO('yolov8n.pt')

    def update(self, image):
        bbox = []

        # detect and track only the people (class 0 in COCO dataset)
        results = self.model.track(source=image, persist=True, tracker="botsort.yaml",classes=0)

        # for r in results:
        boxes = results[0].boxes.xyxy
        for i,box in enumerate(boxes):
            sublist = [] 
            id = results[0].boxes.id[i]
            print(box)
            # get box coordinates in (left, top, right, bottom) format
            # convert the box type to something "digestible", not floats 
            x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())  
            bbox.append([x1,y1,x2,y2,id])
            #sublist.append(id)
            #bbox.append(sublist)
            print(bbox)

        return image,bbox

