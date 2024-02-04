################################################################### IMPORTS #################################################################################

from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch

#############################################################################################################################################################


class Tracker():

    """
    Questa classe Tracker gestisce il tracciamento utilizzando il detector YOLOv8l da Ultralytics e DeepSort.
    
    Parametri:
    - filter_class: Classe del filtro (opzionale).
    - gpu: Flag per l'utilizzo della GPU (default: True).
    - shape: Forma del modello (opzionale).

    Attributi:
    - detector: Detector YOLOv8l da Ultralytics.
    - deepsort: Modulo DeepSort per il tracciamento.

    """
    
    def __init__(self, filter_class=None, gpu=True, shape=None):

        # CARICA IL DETECTOR YOLOV8l DA ULTRALYTICS
        self.detector =  YOLO('DETECTOR_models/yolov8l.pt') 
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        print('CONFIG DEEPSORT: ', cfg.DEEPSORT)

        '''
        
        - max_dist: Distanza massima tra le feature per considerare un match. Se la distanza tra le feature di due oggetti è superiore a questo valore, non saranno considerati lo stesso oggetto.
        - min_confidence: Soglia minima per considerare un rilevamento valido. I rilevamenti con una confidence inferiore a questo valore saranno ignorati.
        - nms_max_overlap: Valore massimo di sovrapposizione per la Non-Maximum Suppression (NMS). La NMS è utilizzata per eliminare i rilevamenti ridondanti.
        - max_iou_distance: Distanza massima tra la posizione predetta (con Kalman) della traccia e quella della detection. Se la distanza IoU tra due detection è superiore a questo valore, non saranno considerati lo stesso oggetto.
        - max_age: Età massima di una traccia prima che venga rimossa. Le tracce che non sono state aggiornate per un numero di frame superiore a max_age saranno rimosse.
        - n_init: Numero di frame consecutivi in cui un oggetto deve essere rilevato prima che la sua traccia sia considerata confermata. Le tracce non confermate saranno rimosse dopo n_init frame.
        - nn_budget: Numero massimo di feature da conservare per ciascuna traccia. Se una traccia ha più di nn_budget feature, le feature più vecchi saranno rimossi.
        
        '''

        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, 
                            n_init=cfg.DEEPSORT.N_INIT, 
                            nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=gpu, 
                            shape=shape)
        
        self.filter_class = filter_class

    def update(self, image):

        """
        Aggiorna lo stato del tracker con nuovi frame.

        Parameters:
        - image: L'immagine corrente da elaborare e tracciare.

        Returns:
        - image: L'immagine originale.
        - outputs: Gli output del tracciamento (tracce dei soggetti).
        
        """

        bbox = []
        scores = []

        # RILEVA E TRACCIA SOLO LE PERSONE (CLASSE 0 NEL DATASET COCO)
        results = self.detector(source=image,classes=0, show_labels=False,show_conf=False, show_boxes=False)
        try:
            boxes = results[0].boxes.xywh
            for i,box in enumerate(boxes):
                x1, y1, w, h = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                if h < 80:
                    continue
                scores.append(results[0].boxes.conf[i])
                bbox.append([x1,y1,w,h])
            bbox_xywh = torch.Tensor(bbox)

            # AGGIORNA LO STATO DEL TRACKER
            outputs = self.deepsort.update(bbox_xywh, scores, image)    
            
        except Exception:   
            bbox = []
            outputs = []

        return image, outputs

