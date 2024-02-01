import os
import cv2
from DeepSort_yolov8 import Tracker
import numpy as np
import pickle

det = []
tracker = Tracker()
paths = ["C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-02/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-04/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-05/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-09/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-10/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-11/img1",
         "C:/Users/rosar/Desktop/AV_progetto/MOT16/train/MOT16-13/img1"]
save_paths = [  'MOT16-02.npy',
                'MOT16-04.npy',
                'MOT16-05.npy',
                'MOT16-09.npy',
                'MOT16-10.npy',
                'MOT16-11.npy',
                'MOT16-13.npy']
for i,path in enumerate(paths):
    tracker.deepsort.det = []
    tracker.deepsort.frame = 0
    for img_path in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_path))
        tracker.update(img)
        #det.append(tracker.deepsort.det)
    print('\n\nSAVING\n\n',save_paths[i])
    with open(save_paths[i],'wb+') as f:
        pickle.dump(tracker.deepsort.det,f)

            