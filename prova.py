from tracker import Tracker
import cv2 
import time
import cv2
from YOLOX.yolox.data.datasets import COCO_CLASSES as class_names
import argparse
from PIL import Image
from torchvision import transforms as T

GPU = True
tracker = Tracker( model='yolox-s',ckpt='./yolox_s.pth',filter_class=['person'],gpu=GPU)    # instantiate Tracker

######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
# parser.add_argument('image_path', help='Path to test image')
# parser.add_argument('--dataset', default='market', type=str, help='dataset')
# parser.add_argument('--backbone', default='resnet50', type=str, help='model')
# parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()
args.use_id = None



models_path = {'uc_model':'models/best_model_uc_alexnet.pth',
                'lc_model':'models/best_model_lc_alexnet.pth',
                'g_model':'models/best_model_g_alexnet.pth',
                'b_model':'models/best_model_b_alexnet.pth',
                'h_model':'models/best_model_h_alexnet.pth'}
    
from par_model import PARModel
color_labels = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white','yellow']
gender_labels = ['male','female']
binary_labels = ['no', 'yes']
par_model = PARModel(models_path, ('cuda' if GPU else 'cpu'), backbone='alexnet')
par_transforms = T.Compose([
        T.Resize((90,220)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

result_path = 'result.txt'
f = open(result_path, 'w+')
cap = cv2.VideoCapture('test2.mp4')  # open one video
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
        img = cv2.resize(img,(700,600))
        t_start = round(time.time() * 1000)
        img_visual, bbox = tracker.update(img.copy())  # feed one frame and get result
        f.write('FRAME: {}\n'.format(i-discard))
        par_start = round(time.time() * 1000)
        for b in bbox:
            #print(b)
            id = b[-1]
            # x1 = (b[0] - 5 if b[0]-5 >= 0 else b[0])
            # y1 = (b[1] - 5 if b[1]-5 >= 0 else b[1])
            # x2 = b[2] + 5
            # y2 = b[3] + 5
            x1, y1, x2, y2 = b[:-1] 
            id = b[-1]
            ###################
            # img_prova = img[y1:y2,x1:x2]
            # cv2.imshow('test',img_prova)
            # cv2.waitKey(0)
            patch = img[y1:y2,x1:x2].copy()
            #print(patch.shape)
            src = Image.fromarray(patch)
            src = par_transforms(src).unsqueeze(dim=0)
            if GPU:
                src = src.to('cuda')
                
            if not args.use_id:
                pred_uc, pred_lc, pred_g, pred_b, pred_h = par_model.predict(src)
            else:
                pred_uc, pred_lc, pred_g, pred_b, pred_h = par_model.predict(src)
            #pred = torch.gt(out, torch.ones_like(out)*9/20 )  # threshold=0.5
            
            res = {'upper_color: ': color_labels[pred_uc.argmax(dim=1).item()],'lower_color: ' : color_labels[pred_lc.argmax(dim=1).item()],
                 'gender: ' : gender_labels[pred_g.argmax(dim=1).item()], "bag: " : binary_labels[pred_b.argmax(dim=1).item()],
                 "hat: " : binary_labels[pred_h.argmax(1).item()]}
            
            #res = Dec.decode(pred)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            txt_size = cv2.getTextSize('Person' + str(id), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img,'Person' + str(id),(x1+1,y1 + txt_size[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
            for k,t in enumerate(res.keys()):
                #print(k, t + res[t])
                txt_size = cv2.getTextSize(t + res[t], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(img,t+res[t],(x1+1,y1+((k+1)*10) + txt_size[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0), thickness=1)
            
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