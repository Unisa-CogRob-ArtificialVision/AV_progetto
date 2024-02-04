from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0
from torchvision.models import vgg11, vgg13, alexnet, resnet50
from torchvision.models import swin_s
import torch
from time import time
from tqdm import tqdm


device = 'cuda'                                           
dim = 10    
data = []

models = {'resnet':resnet18(pretrained=True), 
          'mobilenet_v2':mobilenet_v2(pretrained=True),
          'shufflenet_v2':shufflenet_v2_x1_0(pretrained=True), 
          'squeezenet1':squeezenet1_0(pretrained=True),
          'vgg11': vgg11(pretrained=True),
          'vgg13': vgg13(pretrained=True),
          'alexnet': alexnet(pretrained=True),
          'resnet50': resnet50(pretrained=True),
          'swin_s': swin_s(pretrained=True)}


for m in models.keys():
    models[m].to(device)

results = [0]*len(models)

for i,m in enumerate(tqdm(models.keys())):

    for _ in range(dim):
        x = torch.randn(1,3,90,220).to(device)
        s = time()
        out = models[m](x)
        out = models[m](x)
        out = models[m](x)
        out = models[m](x)
        out = models[m](x)
        e = time() - s
        results[i] += e

for i,m in enumerate(models.keys()):
    print('model', m, ":", results[i]/dim)
