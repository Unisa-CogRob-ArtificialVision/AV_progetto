####### imports
import cv2
import numpy as np
import argparse
import json
from time import time
import math

from tracker import Tracker
from YOLOX.yolox.data.datasets import COCO_CLASSES as class_names

import torch
from torchvision import transforms as T
from PIL import Image
from par_model import PARModel


###################################################################################################### utility functions
def insert_roi_sensors(img, roi):
    for r in roi.keys():
        x = roi[r]['x']
        y = roi[r]['y']
        width = roi[r]['width']
        height = roi[r]['height']
        h, w, _ = img.shape
        x1, y1 = int(x), int(y)
        x2, y2 = int(width + x1), int(height + y1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
    return img

##### convenzione bb -> [x, y, w, h]    (x,y) coordinate angolo superiore sinistro, (w,h) larghezza e altezza
def get_bb_center(bb):
    center_x = (bb[0] + bb[2])/2
    center_y = (bb[1] + bb[3])/2
    return center_x, center_y

##### convenzione bb -> [x, y, w, h]    (x,y) coordinate angolo superiore sinistro, (w,h) larghezza e altezza
def bb_in_roi(bb, roi):
    for r in roi.keys():    # per ogni roi
        x = roi[r]['x']                   
        y = roi[r]['y']
        width = roi[r]['width']
        height = roi[r]['height']
        
        # calcolo centro bb
        center_x , center_y = get_bb_center(bb)
        
        # verifica 
        if (center_x > x and center_x < x+width) and (center_y > y and center_y < y + height):
            return r
    return None # restituisco la roi in cui è presente il bb, altrimenti None
    



def read_config(config_path):
    with open(config_path,'r+') as f:
        data = json.load(f)
        return data


def draw_bbox(img, bb, par_data, id):
    x1, y1 = bb[0], bb[1]
    x2, y2 = bb[2], bb[3]
    cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0))
    id_size = cv2.getTextSize('Person ' + str(id), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(img, 'Person ' + str(id), (x1+1, y1 + id_size[1]), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
    for i,attr in enumerate(par_data.keys()):
        txt_size = cv2.getTextSize(attr+": "+par_data[attr], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(img, attr+": "+par_data[attr], (x1+1, y1+((i+1)*10) + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
    return img

def parse_par_pred(preds, color_labels, gender_labels, binary_labels):
    pred_uc, pred_lc, pred_g, pred_b, pred_h = preds[0], preds[1], preds[2], preds[3], preds[4]
    uc_label = color_labels[pred_uc.argmax(dim=1).item()]
    lc_label = color_labels[pred_lc.argmax(dim=1).item()]
    g_label = gender_labels[pred_g.argmax(dim=1).item()]
    b_label = binary_labels[pred_b.argmax(dim=1).item()]
    h_label = binary_labels[pred_h.argmax(dim=1).item()]
    return {'upper_color':uc_label,'lower_color': lc_label,'gender': g_label,'bag': b_label,'hat': h_label}

###################################################################################################### read configs
# roi1 = {'x': 0.1,'y':0.1,'width':0.4,'height':0.4}
# roi2 = {'x': 0.5,'y':0.7,'width':0.5,'height':0.3}
# roi = {'roi1': roi1, 'roi2': roi2}

GPU = True
if GPU:
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--video",default="C://Users//rosar//Desktop//test_video.mp4", type=str)
parser.add_argument("--configuration",default="config.txt", type=str)
parser.add_argument("--results",default="results.txt", type=str)
args, _ = parser.parse_known_args()

print('WORKING WITH ARGS:',args)

video_path = args.video
roi = read_config(args.configuration)
results_path = args.results

processing_height = 600
processing_width = 800
for r in roi.keys():
    roi[r]['x'] *= processing_width
    roi[r]['y'] *= processing_height
    roi[r]['width'] *= processing_width
    roi[r]['height'] *= processing_height
#print('ROI:',"\n",roi)
###################################################################################################### load tracker/detector
tracker = Tracker( model='yolox-s',ckpt='./yolox_s.pth',filter_class=['person'],gpu=GPU)    # instantiate Tracker

###################################################################################################### load par model
models_path = {'uc_model':'models/best_model_uc_alexnet.pth',
                'lc_model':'models/best_model_lc_alexnet.pth',
                'g_model':'models/best_model_g_alexnet.pth',
                'b_model':'models/best_model_b_alexnet.pth',
                'h_model':'models/best_model_h_alexnet.pth'}
    
color_labels = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white','yellow']
gender_labels = ['male','female']
binary_labels = ['no', 'yes']
task_label_pairs = {'upper_color': color_labels,
         'lower_color': color_labels,
         'gender': gender_labels,
         'bag': binary_labels,
         'hat': binary_labels}
par_model = PARModel(models_path, device, backbone='alexnet')
par_transforms = T.Compose([
        T.Resize((90,220)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
###################################################################################################### load video
cap = cv2.VideoCapture(video_path)

###################################################################################################### start application

bbox_test = [[0,0,10,10, 1], [100,100,130,130, 2], [200,200,240,250, 3], [1000,900,1050,950, 4]]
tracking_id = {}                # contiene le persone tracciate dall'algoritmo (con tutti gli id e le info)
results_json = {"people": []}   # conitene l'output file
additional_info = {}            # contiene informazioni aggiuntive sulle persone tracciate
count_struct = {}               # contiene per ogni label quante volte è stata classificata come tale, per ogni id


_, img = cap.read()
skip_frame = 5
fps = cap.get(cv2.CAP_PROP_FPS)
frame = 0
while img is not None:
    ss = time()
    frame += 1

    #par_data = []
    updated_id = []
    ##### tracking -> outputs the bounding_boxes
    if frame % skip_frame == 0:
        final_img = cv2.resize(img, (processing_width,processing_height)).copy()
        s = time()
        ### track   -> outputs the bounding box
        img_visual, bbox = tracker.update(final_img.copy())
        e = time() - s
        print('TRACK time:',e)
        
        s = time()
        for i,bb in enumerate(bbox):
            #print('person detected')
            
            id = int(bb[-1])
            x1, y1, x2, y2 = bb[:-1]

            
            #print(i,"presence:",bb_in_roi(bb,roi))
            occupied_roi = bb_in_roi(bb,roi)
            if occupied_roi is not None or True:
                #print('person in roi',occupied_roi)
                # x1, y1 = bb[0], bb[1]
                # x2, y2 = x1 + bb[2], y1 + bb[3]

                # x1, y1 = bb[0], bb[1]
                # x2, y2 = bb[2], bb[3]

                
                ######### test values
                # id = i
                
                # pred_uc, pred_lc, pred_g, pred_b, pred_h = [],[],[],[],[] 
                # if frame > 0:
                #     par_data = {'upper_color':'blue', 'lower_color':'black', 'gender':'male', 'bag':'no', 'hat': 'no'}
                # if frame > 10:
                #     par_data = {'upper_color':'orange', 'lower_color':'yellow', 'gender':'female', 'bag':'yes', 'hat': 'yes'}
                # if frame > 30:
                #     par_data = {'upper_color':'green', 'lower_color':'red', 'gender':'male', 'bag':'no', 'hat': 'no'}
                ##########
                    

                patch = par_transforms(Image.fromarray(final_img[y1:y2,x1:x2].copy())).unsqueeze(0).to(device)
                pred_uc, pred_lc, pred_g, pred_b, pred_h = par_model.predict(patch)
                par_data = parse_par_pred([pred_uc, pred_lc, pred_g, pred_b, pred_h], color_labels, gender_labels, binary_labels)
                roi_data = {'roi': occupied_roi}
                
                ### update json file
                if id not in tracking_id.keys():
                    tracking_id[id] = {}
                    person = {'id':id}
                    person.update({'roi1_passages':0, 'roi1_persistence_time':0,'roi2_passages':0, 'roi2_persistence_time':0})
                    additional_info[id] = {'current_roi': None, 'frame_count_roi1': 0, 'frame_count_roi2': 0, 'index': None}
                else:
                    person = tracking_id[id]    # modificare con l'aggiunta di una scelta basata sulla media dei frame (per par)
                
                ## update par data info
                if id not in count_struct.keys():
                    count_struct[id] = {}
                    for task in par_data.keys():
                        count_struct[id][task] = {}
                        label = task_label_pairs[task]
                        for l in label:
                            count_struct[id][task][l] = 0
                    
                for task in par_data.keys():
                    count_struct[id][task][par_data[task]] += 1

                for task in count_struct[id]:
                    max = -1
                    for l in count_struct[id][task]:
                        if count_struct[id][task][l] > max:
                            max_label = l
                            max = count_struct[id][task][l]
                    par_data[task] = max_label
                    
                    
                person.update(par_data)
                ## update roi info
                if occupied_roi == 'roi1':
                    roi_passages = 'roi1_passages'
                    roi_persistence_time = 'roi1_persistence_time'
                    frame_count_roi = 'frame_count_roi1'
                else:
                    roi_passages = 'roi2_passages'
                    roi_persistence_time = 'roi2_persistence_time'
                    frame_count_roi = 'frame_count_roi2'
                
                if additional_info[id]['current_roi'] == None or occupied_roi != additional_info[id]['current_roi']:
                    additional_info[id]['current_roi'] = occupied_roi
                    person[roi_passages] += 1

                additional_info[id][frame_count_roi] += 1
                person[roi_persistence_time] = math.floor(additional_info[id][frame_count_roi]*skip_frame/fps)  # da cambiare con il conteggio dei secondi, questi sono soglo gli fps 
                
                ## save changes for the next iteration
                updated_id.append(id)
                tracking_id[id] = person
                # print('TYPES')
                # for k in person:
                #     print(k, person[k], type(person[k]))

                ## update results json
                if additional_info[id]['index'] == None:
                    additional_info[id]['index'] = len(results_json['people'])
                    results_json['people'].append(person)
                else:
                    idx = additional_info[id]['index']
                    results_json['people'][idx] = person                    
                

                final_img = draw_bbox(final_img, bb, par_data, i)
        final_img = insert_roi_sensors(final_img, roi) 
        e = time() - s
        print('PAR time:',e)    
            #print("\nUPDATED ID",updated_id)
        for id in additional_info.keys():
            #print(id, updated_id)
            if id not in updated_id:
                additional_info[id]['current_roi'] = None

        #json_object = json.dumps(count_struct, indent = 3)
        #json_object = json.dumps(results_json['people'], indent = 3) 
        #print(json_object)
        cv2.imshow('test_roi',final_img)
        cv2.waitKey(1)
    _, img = cap.read()
    # e = time() - ss
    # print('other time',e)