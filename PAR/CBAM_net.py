import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import torch
from CBAM2 import CBAM


class CBAMnet(nn.Module):
    """
    A Multi-Task Attention Network that performs classification tasks on multiple attributes
    of an input image. It utilizes a pre-trained ResNet model as a feature extractor and applies
    CBAM attention followed by separate classifiers for each task.
    """
    def __init__(self):
        super(CBAMnet, self).__init__()
        
        # Load the pretrained ResNet model, excluding the final classification layer.
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.cbam = nn.ModuleDict({
            'upper_color': CBAM(2048,reduction_ratio=8,kernel_size=7),
            'lower_color': CBAM(2048,reduction_ratio=8,kernel_size=7),
            'gender': CBAM(2048,reduction_ratio=8,kernel_size=7),
            'bag': CBAM(2048,reduction_ratio=8,kernel_size=7),
            'hat': CBAM(2048,reduction_ratio=8,kernel_size=7),
        })


        self.conv = nn.ModuleDict({
            "upper_color": nn.Sequential(
                nn.Conv2d(2048,128,7,padding=int((7-1)/2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),),
            "lower_color":nn.Sequential(
                nn.Conv2d(2048,128,7,padding=int((7-1)/2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),),
            "gender":nn.Sequential(
                nn.Conv2d(2048,128,7,padding=int((7-1)/2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),),
            "bag":nn.Sequential(
                nn.Conv2d(2048,128,7,padding=int((7-1)/2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),),
            "hat":nn.Sequential(
                nn.Conv2d(2048,128,7,padding=int((7-1)/2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,7,padding= int((7-1)/2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),),
        })

        # Normalizzazione dei batch dopo i moduli di attenzione.
        self.fc = nn.ModuleDict({
            'upper_color': nn.Sequential(
                nn.Linear(64,11)
            ),
            'lower_color': nn.Sequential(
                nn.Linear(64,11)
            ),
            'gender': nn.Sequential(
                nn.Linear(64,2)
            ),
            'bag': nn.Sequential(
                nn.Linear(64,2)
            ),
            'hat': nn.Sequential(
                nn.Linear(64,2)
            ),
        })


    def forward(self, x):
        # Estrai le caratteristiche dal modello base.
        features = self.feature_extractor(x)
        # Inizializzazione del dizionario per gli output.
        outputs = {}

        # Per ogni attributo, applica l'attenzione e passa le caratteristiche ai classificatori.
        for task in self.cbam.keys():
            attn_output = self.cbam[task](features)
            attn_output = self.conv[task](attn_output).squeeze()
            # attn_output = attn_output.view(attn_output.size(0), -1)
            # norm_output = self.batch_norms[task](attn_output)
            # dropped_output = self.dropouts[task](norm_output)
            outputs[task] = self.fc[task](attn_output)

        # outputs["gender"] = outputs["gender"].squeeze(-1)
        # outputs["bag"] = outputs["bag"].squeeze(-1)
        # outputs["hat"] = outputs["hat"].squeeze(-1)

        return outputs
