import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from par_train import SimpleModel, CustomDataset
from matplotlib import pyplot as plt
import os
import sys
import time

from PIL import Image
from tqdm import tqdm



class PARModel():
    def __init__(self, models_path, device, backbone=['resnet']*5):

        self.device = device
        # model = resnet18(pretrained=True)
        # self.backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
        # self.backbone.requires_grad = False
        model_uc = SimpleModel(11,backbone[0])
        checkpoint = torch.load(models_path['uc_model'])
        model_uc.load_state_dict(checkpoint['model_state_dict'])
        # self.uc_head = model_uc.backbone.fc
        model_uc.requires_grad = False
        model_lc = SimpleModel(11,backbone[1])
        checkpoint = torch.load(models_path['lc_model'])
        model_lc.load_state_dict(checkpoint['model_state_dict'])
        # self.lc_head = model_lc.backbone.fc
        model_lc.requires_grad = False
        model_g = SimpleModel(2,backbone[2])
        checkpoint = torch.load(models_path['g_model'])
        model_g.load_state_dict(checkpoint['model_state_dict'])
        # self.g_head = model_g.backbone.fc
        model_g.requires_grad = False
        model_b = SimpleModel(2,backbone[3])
        checkpoint = torch.load(models_path['b_model'])
        model_b.load_state_dict(checkpoint['model_state_dict'])
        # self.b_head = model_b.backbone.fc
        model_b.requires_grad = False
        model_h = SimpleModel(2,backbone[4])
        checkpoint = torch.load(models_path['h_model'])
        model_h.load_state_dict(checkpoint['model_state_dict'])
        # self.h_head = model_h.backbone.fc
        model_h.requires_grad = False

        self.model_uc = model_uc; self.model_uc.to(device); self.model_uc.eval()
        self.model_lc = model_lc; self.model_lc.to(device); self.model_lc.eval()
        self.model_g = model_g; self.model_g.to(device); self.model_g.eval()
        self.model_b = model_b; self.model_b.to(device); self.model_b.eval()
        self.model_h = model_h; self.model_h.to(device); self.model_h.eval()

        # self.backbone.to(device); self.backbone.eval()
        # self.uc_head.to(device); self.uc_head.eval()
        # self.lc_head.to(device); self.lc_head.eval()
        # self.g_head.to(device); self.g_head.eval()
        # self.b_head.to(device); self.b_head.eval()
        # self.h_head.to(device); self.h_head.eval()

        # out11 = self.backbone(x).squeeze()
        # backbone_uc = torch.nn.Sequential(*(list(model_uc.backbone.children())[:-1]))
        # backbone_lc = torch.nn.Sequential(*(list(model_lc.backbone.children())[:-1]))
        # out12 = self.uc_head(out11)
        # out13 = self.lc_head(out11)
        # out2 = backbone_uc(x).squeeze()
        # out3 = backbone_lc(x).squeeze()
        # print(out11.shape, out2.shape)
        # print(out2,"\n",out3)
        
        
    def predict(self,x):
        out_uc = self.model_uc(x)
        out_lc = self.model_lc(x)
        out_g = self.model_g(x)
        out_b = self.model_b(x)
        out_h = self.model_h(x)
        return [out_uc,out_lc,out_g,out_b,out_h]

if __name__ == '__main__':
    models_path = {'uc_model':'models/best_model_uc_vgg11.pth',
                'lc_model':'models/best_model_lc_vgg11.pth',
                'g_model':'models/best_model_gender_vgg11.pth',
                'b_model':'models/best_model_bag_vgg11.pth',
                'h_model':'models/best_model_hat_vgg11.pth'}

    color_labels = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white','yellow']
    gender_labels = ['male','female']
    binary_labels = ['no', 'yes']


    transform = transforms.Compose([
        transforms.Resize((90,220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_batch_size = 1
    val_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=None)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
    #val_set, _ =  torch.utils.data.random_split(val_set,[0.3,0.7])
    print('\nDATASET done\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); 
    print('\nLOADING MODEL\n')
    model = PARModel(models_path, device)   
    print('\nMODEL done\n')
    print("\nDEVICE:",device)
    print('\nSTARTING\n')
    #model.eval()
    c_uc, c_lc, c_g, c_b, c_h = 0, 0, 0, 0, 0
    cl_uc, cl_lc, cl_g, cl_b, cl_h = 0, 0, 0, 0, 0

    cl_uc2 = 0

    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(device), labels.squeeze((0)).to(device)
        s = time.time()
        pred_uc, pred_lc, pred_g, pred_b, pred_h = model.predict(inputs)
        e = time.time() - s
        # print(e)
        # print("uc",pred_uc.argmax(dim=1).item()," --- ", labels[0].item())
        # print("lc",pred_lc.argmax(dim=1).item()," --- ", labels[1].item())
        # print("g", pred_g.argmax(dim=1).item()," --- ", labels[2].item())
        # print("b", pred_b.argmax(dim=1).item()," --- ", labels[3].item())
        # print("h", pred_h.argmax(dim=1).item()," --- ", labels[4].item())
        # print('\n') 
        # if pred_uc.argmax(dim=1).item() != 0:
        #     print('YEES')
        #print(labels)
        if labels[0].item() >= 0:
            cl_uc += pred_uc.argmax(dim=1) == labels[0]
            cl_uc2 += pred_uc.argmax(dim=1).item() == labels[0].item()
            c_uc += 1
        if labels[1].item() >= 0:
            cl_lc += (pred_lc.argmax(dim=1).item() == labels[1].item())
            c_lc += 1
        if labels[2].item() >= 0:
            cl_g += (pred_g.argmax(dim=1).item() == labels[2].item())
            c_g += 1
        if labels[3].item() >= 0:
            cl_b += (pred_b.argmax(dim=1).item() == labels[3].item())
            c_b += 1
        if labels[4].item() >= 0:
            cl_h += (pred_h.argmax(dim=1).item() == labels[4].item())
            c_h += 1

        # plt.imshow(inputs.squeeze(0).numpy().transpose((1,2,0)))
        # plt.text(10,10,'upper_color: ' + color_labels[pred_uc.argmax(dim=1).item()] + '\nlower_color: ' + color_labels[pred_lc.argmax(dim=1).item()]
        #          + '\ngender: ' + gender_labels[pred_g.argmax(dim=1).item()] + "\n bag: " + binary_labels[pred_b.argmax(dim=1).item()]
        #          + "\nhat: " + binary_labels[pred_h.argmax(1).item()])
        # plt.show()

        
    print(cl_uc.item()/c_uc, cl_uc2/c_uc)
    print(cl_lc/c_lc)
    print(cl_g/c_g)
    print(cl_b/c_b)
    print(cl_h/c_h)