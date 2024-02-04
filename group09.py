################################################################### IMPORTS #################################################################################

import cv2
import argparse
import json
from time import time
import math
import sys
import torch
from DeepSort_yolov8 import Tracker
from torchvision import transforms as T
from PIL import Image
from PAR.par_model import PARModel

#############################################################################################################################################################

##################################################################### UTILITY FUNCTIONS #####################################################################

def insert_roi_sensors(img, roi):

    """
    Questa funzione inserisce rettangoli che rappresentano regioni di interesse (ROI) in un'immagine,
    e aggiunge etichette testuali indicando l'indice di ciascuna ROI.

    Parameters:
    - img: L'immagine di input.
    - roi: Un dizionario che contiene informazioni su ciascuna ROI, inclusi le coordinate del punto più in alto a sinistra (x, y),
        la larghezza e l'altezza.

    Returns:
    - img: L'immagine di input con i rettangoli che rappresentano le ROI e le etichette corrispondenti.
    """

    for i,r in enumerate(roi.keys()):
        x = roi[r]['x']
        y = roi[r]['y']
        width = roi[r]['width']
        height = roi[r]['height']
        x1, y1 = int(x), int(y)
        x2, y2 = int(width + x1), int(height + y1)
        id_size = cv2.getTextSize(str(i+1), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        cv2.putText(img, str(i+1), (x1+5, y1+5 + id_size[1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), thickness=2)    
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=2) 

    return img

def get_bb_center(bb):

    """Restituisce il centro del bounding box.
    
    convenzione bb -> [x1, y1, x2, y2]    
    (x1,y1) coordinate angolo superiore sinistro, 
    (x2,y2) coordinate angolo inferiore destro.

    Parameters:
    - bb: Lista che rappresenta il bounding box nel formato [x1, y1, x2, y2].

    Returns:
    - center_x: Coordinata x del centro del bounding box.
    - center_y: Coordinata y del centro del bounding box.
    
    """

    center_x = (bb[0] + bb[2])/2
    center_y = (bb[1] + bb[3])/2

    return center_x, center_y

def bb_in_roi(bb, roi):

    """
    Controlla se il bounding box è nella regione di interesse (ROI).

    Parameters:
    - bb: Lista che rappresenta il bounding box nel formato [x1, y1, x2, y2].
    - roi: Dizionario che contiene informazioni sulle regioni di interesse.

    Returns:
    - La chiave della ROI in cui si trova il bounding box se presente, altrimenti None.
    
    """

    for r in roi.keys():    
        x = roi[r]['x']                   
        y = roi[r]['y']
        width = roi[r]['width']
        height = roi[r]['height']
        
        # CALCOLO DEL CENTRO DEL BB
        center_x , center_y = get_bb_center(bb)
        
        # VERIFICA - RESTITUISCE LA ROI IN CUI E' PRESENTE IL BB, ALTRIMENTI NONE
        if (center_x > x and center_x < x+width) and (center_y > y and center_y < y + height):
            return r
        
    return None 
    
def read_config(config_path):

    """
    Legge il file di configurazione delle ROI e restituisce il contenuto.

    Parameters:
    - config_path: Percorso del file di configurazione delle ROI.

    Returns:
    - data: Contenuto del file di configurazione delle ROI.

    """

    with open(config_path,'r+') as f:
        data = json.load(f)

        return data


def draw_bbox(img, bb, par_data, id, color, par=True):

    """
    Disegna il bounding box nell'immagine e inserisce le informazioni di tracking (id) e i dati di Pedestrian Attribute Recognition (PAR).

    Parameters:
    - img: L'immagine su cui disegnare il bounding box.
    - bb: Lista che rappresenta il bounding box nel formato [x1, y1, x2, y2].
    - par_data: Dati di Pedestrian Attribute Recognition (PAR).
    - id: Identificatore del tracking.
    - color: Colore del bounding box e del testo.
    - par: Booleano che indica se inserire i dati di PAR o meno (impostato di default su True).

    Returns:
    - img: L'immagine con il bounding box disegnato e le informazioni di tracking e PAR inserite.

    """

    x1, y1 = int(bb[0]), int(bb[1])
    x2, y2 = int(bb[2]), int(bb[3])
    cv2.rectangle(img,(x1, y1), (x2, y2), color, thickness=2)
    id_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    cv2.rectangle(img, (x1+3,y1+3),(x1 + id_size[0]+3, y1 + id_size[1]+6), (255,255,255), thickness=-1)
    cv2.putText(img,str(id), (x1+3, y1+3 + id_size[1]), cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness=1)

    if par:
        tot_size_x = 115
        tot_size_y = 70
        cv2.rectangle(img, (x1+1,y2+5),(x1+6+tot_size_x,y2 + tot_size_y), (255,255,255),thickness=-1)

        for i,attr in enumerate(par_data.keys()):
            txt_size = cv2.getTextSize(attr+": "+str(par_data[attr]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img, attr+": "+str(par_data[attr]), (x1+3, y2+((i+1)*10) + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 0), thickness=1)

    return img

def parse_par_pred(preds, color_labels, gender_labels, binary_labels):

    """
    Prende in input i tensori di predizione del modello PAR (gli output della .predict()) e restituisce la label associata alla predizione.

    Parameters:
    - preds: Tuple di tensori di predizione del modello PAR (output della .predict()).
    - color_labels: Lista delle etichette per la classificazione dei colori.
    - gender_labels: Lista delle etichette per la classificazione del genere.
    - binary_labels: Lista delle etichette binarie.

    Returns:
    - Un dizionario contenente le label associate alla predizione per le varie caratteristiche PAR.
    
    """

    pred_uc, pred_lc, pred_g, pred_b, pred_h = preds[0], preds[1], preds[2], preds[3], preds[4]
    uc_label = color_labels[pred_uc.argmax(dim=1).item()]
    lc_label = color_labels[pred_lc.argmax(dim=1).item()]
    g_label = gender_labels[pred_g.argmax(dim=1).item()]
    b_label = binary_labels[pred_b.argmax(dim=1).item()]
    h_label = binary_labels[pred_h.argmax(dim=1).item()]

    return {'upper_color':uc_label,'lower_color': lc_label,'gender': g_label,'bag': b_label,'hat': h_label}

################################################################### READ CONFIGS ############################################################################

# CREAZIONE DI UN OGGETTO ARGUMENTPARSER CHIAMATO PARSER PER GESTIRE GLI ARGOMENTI DA RIGA DI COMANDO
parser = argparse.ArgumentParser()
parser.add_argument("--video",default="Example.mp4", type=str)
parser.add_argument("--configuration",default="config.txt", type=str)
parser.add_argument("--results",default="results.txt", type=str)
parser.add_argument("--gpu", default=True, type=bool)

# ESECUZIONE DEL PARSING DEGLI ARGOMENTI DALLA RIGA DI COMANDO, RESTITUENDO ARGS (ARGOMENTI ANALIZZATI) E _ (ARGOMENTI NON ANALIZZATI)
args, _ = parser.parse_known_args()

print('WORKING WITH ARGS:',args)

# VERIFICA SE UTILIZZARE LA GPU O LA CPU IN BASE ALL'ARGOMENTO --GPU SPECIFICATO
GPU = args.gpu
if GPU:
    device = ('cuda' if torch.cuda.is_available() else 'cpu') 
else:
    device = 'cpu'

print('WORKING WITH DEVICE:',device)
video_path = args.video
roi = read_config(args.configuration)
results_path = args.results

# ALTEZZA DI ELABORAZIONE
processing_height = 720 

# LARGHEZZA DI ELABORAZIONE
processing_width = 1020

# RISCALAMENTO DELLE COORDINATE DELLE ROI IN BASE ALLE DIMENSIONI DI ELABORAZIONE
for r in roi.keys():
    roi[r]['x'] *= processing_width
    roi[r]['y'] *= processing_height
    roi[r]['width'] *= processing_width
    roi[r]['height'] *= processing_height

#############################################################################################################################################################
    
################################################################### LOAD TRACKER/DETECTOR ###################################################################

tracker = Tracker(gpu=GPU, shape=(processing_width,processing_height))    

#############################################################################################################################################################

###################################################################### LOAD PAR MODELS ######################################################################

models_path = {'uc_model':'PAR/PAR_models/best_model_uc_alexnet_batch_mod_asym_mod_MIGLIORE.pth',
                'lc_model':'PAR/PAR_models/best_model_lc_alexnet_batch_mod_asym_mod_v2_continue_MIGLIORE.pth',
                'g_model':'PAR/PAR_models/best_model_g_alexnet_batch_mod_asym_mod_MIGLIORE.pth',
                'b_model':'PAR/PAR_models/best_model_b_alexnet_batch_mod_asym_MIGLIORE.pth',
                'h_model':'PAR/PAR_models/best_model_h_alexnet_batch_mod_asym_mod_v2_MIGLIORE.pth'}
    
color_labels = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white','yellow']
gender_labels = ['male','female']
binary_labels = [False, True]
task_label_pairs = {'upper_color': color_labels,
                    'lower_color': color_labels,
                    'gender': gender_labels,
                    'bag': binary_labels,
                    'hat': binary_labels}

par_model = PARModel(models_path, device, backbone=['alexnet']*5)
par_transforms = T.Compose([
                    T.Resize((90,220)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

######################################################################## LOAD VIDEO #########################################################################

cap = cv2.VideoCapture(video_path)

#############################################################################################################################################################

##################################################################### START APPLICATION #####################################################################

tracking_id = {}                # CONTIENE LE PERSONE TRACCIATE DALL'ALGORITMO (CON TUTTI GLI ID E LE INFO)
results_json = {"people": []}   # CONTIENE L'OUTPUT FILE
additional_info = {}            # CONTIENE INFORMAZIONI AGGIUNTIVE SULLE PERSONE TRACCIATE
count_struct = {}               # CONTIENE PER OGNI LABEL QUANTE VOLTE È STATA CLASSIFICATA COME TALE, PER OGNI ID

gui_upper_left = {'People in ROI': 0,
                  'Total persons': 0,
                  'Passages in ROI 1': 0,
                  'Passages in ROI 2': 0}

_, img = cap.read()

if img is None:
    print('Video not found')
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
fps_target = 6

# SKIP_FRAME BASATO SU FPS_TARGET E GLI FPS DEL VIDEO DI INPUT
skip_frame = int(fps/fps_target)        
print('USING SKIP FRAME:', skip_frame)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
frame = 0
start_time = time()

while img is not None:
    ss = time()
    frame += 1
    updated_id = []

    if frame % skip_frame == 0 or frame >= frames:
        final_img = cv2.resize(img, (processing_width,processing_height)).copy()
        s = time()

        ### TRACK -> RESTITUISCE LE BOUNDING BOX
        _, bbox = tracker.update(final_img.copy())
        e = time() - s
        print('TRACK time:',e)
        
        gui_upper_left['People in ROI'] = 0
        gui_upper_left['Total persons'] = len(bbox)
        s = time()

        for i,bb in enumerate(bbox):

            
            id = int(bb[4])
            x1, y1, x2, y2 = bb[:4]

            occupied_roi = bb_in_roi(bb,roi)
            if occupied_roi is not None:
                gui_upper_left['People in ROI'] += 1
            if occupied_roi is None:
                occupied_roi = 'outside_roi'
        
            h, w, _ = final_img.shape      
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # LA PATCH VIENE TRASFORMATA IN TENSORE, NORMALIZZATA SECONDO LE CARATTERISTICHE RICHIESTE DAI MODELLI PER IL PAR E RIDIMENSIONATA A 90x220
            patch = par_transforms(Image.fromarray(final_img[y1:y2,x1:x2].copy())).unsqueeze(0).to(device)
                
            pred_uc, pred_lc, pred_g, pred_b, pred_h = par_model.predict(patch)
            par_data = parse_par_pred([pred_uc, pred_lc, pred_g, pred_b, pred_h], color_labels, gender_labels, binary_labels)
            
            # AGGIORNA IL FILE JSON 
            if id not in tracking_id.keys():
                tracking_id[id] = {}
                person = {'id':id}
                person.update({'roi1_passages':0, 'roi1_persistence_time':0,'roi2_passages':0, 'roi2_persistence_time':0})
                additional_info[id] = {'current_roi': None, 'frame_count_roi1': 0, 'frame_count_roi2': 0, 'index': None, 'last_seen': None}

            else:
                person = tracking_id[id]    
            
            # AGGIORNAMENTO DELLE INFORMAZIONI DEL PAR
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

            # AGGIORNAMENTO DELLE INFORMAZIONI DELLA ROI
            if occupied_roi == 'roi1':
                roi_passages = 'roi1_passages'
                roi_persistence_time = 'roi1_persistence_time'
                frame_count_roi = 'frame_count_roi1'
                gul_roi = 'Passages in ROI 1'
            elif occupied_roi == 'roi2':
                roi_passages = 'roi2_passages'
                roi_persistence_time = 'roi2_persistence_time'
                frame_count_roi = 'frame_count_roi2'
                gul_roi = 'Passages in ROI 2'

            print(id, occupied_roi)

            '''

            La condizione if occupied_roi is not None: verifica se la persona tracciata si trova all'interno di una ROI.
            La condizione successiva:
            Controlla se la persona è appena entrata nella ROI oppure se è già stata tracciata in precedenza ma è uscita dalla ROI e poi rientrata.
            Se la persona è appena entrata nella ROI oppure se è rientrata dopo essere uscita, si verifica se l'ultima ROI vista è diversa dalla ROI attualmente occupata. 
            In caso affermativo, si aumenta il conteggio dei passaggi nella ROI corrente e si aggiorna il contatore generale dei passaggi (variabile roi_passages).
            Viene quindi aggiornato il registro delle informazioni aggiuntive sulla persona tracciata, includendo l'ultima ROI vista e la ROI corrente in cui si trova attualmente.
                        
            '''

            if occupied_roi is not None:    

                if (additional_info[id]['current_roi'] == None and occupied_roi != additional_info[id]['last_seen']) or \
                   (additional_info[id]['current_roi'] is not None and additional_info[id]['current_roi'] != occupied_roi):
                    
                    if additional_info[id]['last_seen'] == 'outside_roi':
                        print(id, occupied_roi, additional_info[id]['last_seen'])

                    if occupied_roi != 'outside_roi':
                        person[roi_passages] += 1
                        gui_upper_left[gul_roi] += 1
                        
                    additional_info[id]['last_seen'] = occupied_roi
                    additional_info[id]['current_roi'] = occupied_roi
                
                if occupied_roi not in ['outside_roi', None]:
                    additional_info[id][frame_count_roi] += 1

                    if frame >= frames:
                        person[roi_persistence_time] = math.floor((((additional_info[id][frame_count_roi]-1)*skip_frame)+1)/fps)  

                    else:    
                        person[roi_persistence_time] = math.floor(additional_info[id][frame_count_roi]*skip_frame/fps)  
                
            # SALVIAMO LE MODIFICHE PER LA PROSSIMA ITERAZIONE
            updated_id.append(id)
            tracking_id[id] = person

            # AGGIORNIAMO IL FILE JSON DEI RISULTATI
            if additional_info[id]['index'] == None:
                additional_info[id]['index'] = len(results_json['people'])
                results_json['people'].append(person)

            else:
                idx = additional_info[id]['index']
                results_json['people'][idx] = person                    
            
            if occupied_roi == 'roi1':
                color = (255, 0, 0)

            elif occupied_roi == 'roi2':
                color = (0, 255, 0)
                
            else: 
                color = (0, 0, 255)

            final_img = draw_bbox(final_img, bb, par_data, id, color, par=True)
        final_img = insert_roi_sensors(final_img, roi)                         
        e = time() - s
        print('PAR time:',e)   

        for id in additional_info.keys():
            if id not in updated_id:
                additional_info[id]['current_roi'] = None

        # LE INFORMAZIONI SULLA SCENA SONO INSERITE IN ALTO A SINISTRA
        cv2.rectangle(final_img, (0,1), (252,100), (255, 255, 255), thickness=-1)
        for i,k in enumerate(gui_upper_left.keys()):
            txt_size = cv2.getTextSize(k+": "+str(gui_upper_left[k]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(final_img, k+": "+str(gui_upper_left[k]), (3, i*20 + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),thickness=2)

        cv2.imshow('test_roi',final_img)
        cv2.waitKey(1)
    _, img = cap.read()
total_time = time() - start_time
print('VIDEO LENGHT:',round(frame/fps),"seconds")
print('TOTAL TIME:',round(total_time),"seconds")

json_object = json.dumps(results_json, indent = 3)
f = open(results_path, 'w+')
f.write(json_object)
print('RESULTS written to:',results_path)
