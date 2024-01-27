from tracker import Tracker
import cv2 
import time
from detector import Detector
import cv2
from YOLOX.yolox.data.datasets import COCO_CLASSES as class_names
import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model
import numpy as np

GPU = True
tracker = Tracker( model='yolox-s',ckpt='./yolox_s.pth',filter_class=['person'],gpu=GPU)    # instantiate Tracker

######################################################################
# Settings
# ---------

dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),  # 200 x 80
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
# parser.add_argument('image_path', help='Path to test image')
# parser.add_argument('--dataset', default='market', type=str, help='dataset')
# parser.add_argument('--backbone', default='resnet50', type=str, help='model')
# parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()
args.dataset = 'duke'
args.backbone = 'resnet50'
args.use_id = None
assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]

print("MODEL NAME:",model_name)

######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./checkpoints', args.dataset, model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

print('GETTING MODEL')
model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
print('LOADING NETWORK')
model = load_network(model)
if GPU:
    model = model.to("cuda")
model.eval()
print('DONE')
#src = load_image(args.image_path)

######################################################################
######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset='market'):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)
        self.name_of_interest = ['bag','gender','upper-color','lower-color','hat','backpack']

    def decode(self, pred):
        ret = []
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]] and name in self.name_of_interest:
                ret.append('{}: {}'.format(name, chooce[pred[idx]]))
        return ret
Dec = predict_decoder(args.dataset)

result_path = 'result.txt'
f = open(result_path, 'w+')
cap = cv2.VideoCapture('test1.mp4')  # open one video
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
i = discard = 5

kk = mean_par = mean_tracking = 0
start = round(time.time() * 1000)
while True:
    
    _, img = cap.read() # read frame from video
    if img is None:
        break
    if i % discard == 0:
        kk += 1
        # print('working on {}'.format(i-discard))
        img = cv2.resize(img,(700,900))
        t_start = round(time.time() * 1000)
        img_visual, bbox = tracker.update(img.copy())  # feed one frame and get result
        f.write('FRAME: {}\n'.format(i-discard))
        par_start = round(time.time() * 1000)
        for b in bbox:
            id = b[-1]
            x1, y1, x2, y2 = b[:-1]
            id = b[-1]
            ###################
            # img_prova = img[y1:y2,x1:x2]
            # cv2.imshow('test',img_prova)
            # cv2.waitKey(0)
            patch = img[y1:y2,x1:x2].copy()
            #print(patch.shape)
            src = Image.fromarray(patch)
            src = transforms(src).unsqueeze(dim=0)
            if GPU:
                src = src.to('cuda')
                
            if not args.use_id:
                out = model.forward(src)
            else:
                out, _ = model.forward(src)
            pred = torch.gt(out, torch.ones_like(out)*9/20 )  # threshold=0.5
            res = Dec.decode(pred)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            txt_size = cv2.getTextSize('Person' + str(id), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img,'Person' + str(id),(x1+1,y1 + txt_size[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
            for k,t in enumerate(res):
                txt_size = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(img,t,(x1+1,y1+((k+1)*10) + txt_size[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
            
            ####################

            f.write('\tID: {}\n\t position: ({},{})-({},{})\n'.format(id,x1,x2,y1,y2))
            f.write('\tPAR: {}\n'.format(res))
        par_end = round(time.time() * 1000)
        mean_par += (par_end - par_start)
        mean_tracking += (par_start - t_start)
        print('PAR time:',par_end - par_start)
        print('TRACKING time:',par_start - t_start)
        cv2.imshow('demo',img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    i += 1
mean_par /= kk
mean_tracking /= kk
end = round(time.time() * 1000)
seconds = round(frames/fps)
print('ELAPSED TIME: {}, VIDEO LENGHT: {}\n'.format((end-start)/1000,seconds))
print('MEAN PAR TIME: {}, MEAN TRACKING TIME: {}'.format(mean_par,mean_tracking))
f.write('ELAPSED TIME: {}, VIDEO LENGHT: {}\n'.format((end-start)/1000,seconds))

cap.release()
cv2.destroyAllWindows()