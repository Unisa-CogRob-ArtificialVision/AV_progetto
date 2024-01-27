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
def get_persons_patch(input,results,thresh=0.5):
    pp = []
    for i in range(0,len(results['boxes'])):
        if results['scores'][i] > thresh and results['class_ids'][i] == 0:      # 0 sta per 'person'
            pp.append(results['boxes'][i])

    patches = []
    for b in pp:
        x0, x1 = int(b[0]), int(b[2])
        y0, y1 = int(b[1]), int(b[3])
        #print(x0,x1,y0,y1)
        patches.append((input[y0:y1,x0:x1],(x0,y0,x1,y1)))
    return patches

######## LOAD DETECTOR
detector = Detector(model='yolox-s', ckpt='yolox_s.pth', gpu=GPU) # instantiate Detector
######## LOAD RESNET (for PAR)

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
    T.Resize(size=(288, 144)),
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
args.dataset = 'market'
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
    model = torch.nn.DataParallel(model).cuda()
    #model = model.to("cuda")
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

# img = cv2.imread('YOLOX/assets/dog.jpg') 	# load image
# img_or = img.copy()
# result = detector.detect(img) 	# detect targets
# img_visual = result['visual'] 	 # visualized image
# # cv2.imshow('detect', img_visual) # imshow
# # cv2.waitKey(0)

# patch = get_patch(img_or,result['boxes'][:4])
# for i,p in enumerate(patch):
#     cv2.imshow('p' + str(i), p)
# cv2.waitKey(0)
cap = cv2.VideoCapture('test1.mp4')
i = discard = 5
Dec = predict_decoder(args.dataset)
while True:

    _, img = cap.read()
    if img is None:
        break
    if i % discard == 0:
        img_or = img.copy()
        img_or = cv2.resize(img_or,(640,480))
        result = detector.detect(img_or)
        patch = get_persons_patch(img_or,result,0.75)
        for j,p in enumerate(patch):
            imgp = p[0]
            src = Image.fromarray(imgp)
            src = transforms(src)
            src = src.unsqueeze(dim=0)
            if GPU:
                src = src.to('cuda')
            if not args.use_id:
                out = model.forward(src).cpu()
            else:
                out, _ = model.forward(src.cuda()).cpu()
            pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
            res = Dec.decode(pred)
            # cv2.rectangle(img_or, (p[1][0], p[1][1]), (p[1][2], p[1][3]), (0, 255, 0), 2)

            # for k,t in enumerate(res):
            #     txt_size = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            #     cv2.putText(img_or,t,(p[1][0]+1,p[1][1]+(k*10) + txt_size[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=2)
        cv2.imshow('demo',img_or)
        cv2.waitKey(1)
        # cv2.imshow('test',cv2.resize(result['visual'],(1200,700)))
        # cv2.waitKey(1)
        
    i += 1
cap.release()
cv2.destroyAllWindows()

