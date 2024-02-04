from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0
from torchvision.models import vgg11, vgg13, vgg16, vgg19, alexnet, resnet50, resnet101, densenet121, densenet169, inception_v3, googlenet
from torchvision.models import convnext_small, swin_s, convnext_base, convnext_large,  convnext_tiny, efficientnet_b3, maxvit_t, mnasnet0_75, regnet_y_16gf, swin_s, vit_b_32
import torch
from time import time
from tqdm import tqdm
device = 'cuda'                                           ######## <- inserire device di test
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
#### test4
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
