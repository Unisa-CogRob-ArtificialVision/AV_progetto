import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from attention_mechanisms.se_module import SELayer
from attention_mechanisms.simam import simam_module
from .par_train import SimpleModel, CustomDataset
from matplotlib import pyplot as plt
import os
import sys


from PIL import Image
from tqdm import tqdm



class PARModel(nn.Module):
    def __init__(self, models_path):
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad = False
        model = SimpleModel(11)
        checkpoint = torch.load(models_path['uc_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.uc_head = model.fc
        self.uc_head.requires_grad = False
        checkpoint = torch.load(models_path['lc_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.lc_head = model.fc
        self.lc_head.requires_grad = False
        checkpoint = torch.load(models_path['g_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.g_head = model.fc
        self.g_head.requires_grad = False
        checkpoint = torch.load(models_path['b_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.b_head = model.fc
        self.gbhead.requires_grad = False
        checkpoint = torch.load(models_path['h_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        self.h_head = model.fc
        self.h_head.requires_grad = False
        
    def forward(self,x):
        x1 = self.backbone(x)
        out_uc = self.uc_head(x1)
        out_lc = self.lc_head(x1)
        out_g = self.g_head(x1)
        out_b = self.b_head(x1)
        out_h = self.h_head(x1)
        #return torch.cat([out_uc,out_lc,out_g,out_b,out_h],dim=1)
        return [out_uc,out_lc,out_g,out_b,out_h]
        
models_path = {'uc_model':'models/best_model_uc.pth',
               'lc_model':'models/best_model_lc.pth',
               'g_model':'models/best_model_g.pth',
               'b_model':'models/best_model_b.pth',
               'h_model':'models/best_model_h.pth'}


color_labels = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white','yellow']
gender_labels = ['male','female']
binary_labels = ['no', 'yes']


transform = transforms.Compose([
    transforms.Resize((90,220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_set = CustomDataset("validation_set","validation_set.txt", transform)

model = PARModel(models_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE:",device)
model.to(device)

model.eval()
c_uc, c_lc, c_g, c_b, c_h = 0, 0, 0, 0, 0
for inputs, labels in val_set:
    inputs, labels = inputs.to(device), labels.to(device)

    pred_uc, pred_lc, pred_g, pred_b, pred_h = model(inputs)
    
    print("uc",pred_uc.argmax(dim=1) +" --- "+ labels[0])
    print("lc",pred_lc.argmax(dim=1) +" --- "+ labels[1])
    print("g", pred_g.argmax(dim=1)  +" --- "+ labels[2])
    print("b", pred_b.argmax(dim=1)  +" --- "+ labels[3])
    print("h", pred_h.argmax(dim=1)  +" --- "+ labels[4])
    print('\n') 
    
    if labels[0] == 0:
        c_uc += pred_uc.argmax(dim=1) == labels[0]
    

    plt.imshow(inputs.numpy().transpose((1,2,0)))
    plt.text(10,10,'upper_color: ' + color_labels[pred_uc.argmax(dim=1)] + '\nlower_color: ' + color_labels[pred_lc.argmax(dim=1)]
             + '\ngender: ' + gender_labels[pred_g.argmax(dim=1)] + "\n bag: " + binary_labels[pred_b.argmax(dim=1)]
             + "\nhat: " + binary_labels[pred_h.argmax(1)])
    plt.show()

    
