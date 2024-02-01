import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as M
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from torchvision.models import vgg11
import random
import os
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0, alexnet, vgg11, convnext_large, ConvNeXt_Large_Weights, ResNet18_Weights, VGG11_Weights, mobilenet_v3_large
from balanced_sampler import BalancedBatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
# Define a simple CNN model
class SimpleModel(nn.Module):
    def __init__(self,num_classes,model='resnet'):
        super(SimpleModel, self).__init__()

        models = {'resnet': resnet18(pretrained=True), 
                  'shufflenet': shufflenet_v2_x1_0(pretrained=True), 
                  'squeezenet': squeezenet1_0(pretrained=True), 
                  'mobilenet': mobilenet_v2(pretrained=True),
                  'alexnet': alexnet(pretrained=True),
                  'vgg11': vgg11(pretrained=True)
                  }
        if model not in models.keys():
            model = 'resnet'
        
        print('LOADING MODEL: ', model)
        self.backbone = models[model]
        #print(self.backbone)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        if model == 'alexnet' or model == 'vgg11':
            self.backbone.classifier[6] = nn.Linear(4096, num_classes)
            self.backbone.classifier[6].requires_grad = True
            # self.backbone.classifier[4].requires_grad = True
            # self.backbone.classifier[2].requires_grad = True
            # self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            # self.backbone.classifier[1].requires_grad = True
        else:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            self.backbone.fc.requires_grad = True


    def forward(self, x):
        x2 = self.backbone(x)
        return x2

    
    
# Define a custom dataset
class CustomDataset(IterableDataset):
    def __init__(self, data_path, labels_path, transform=None, target_label=None, shuffle=True):
        self.data_dir = data_path
        self.target_label = target_label        # indica la label su cui voglio fare l'addestramento. Serve per scartare solo i dati che hanno "-1" per la label target
        self.task_name = ['upper_color', 'lower_color', 'gender', 'bag', 'hat']

        self.shuffle = shuffle
        self.labels = self._read_labels(labels_path)  
        self.task_min = self._compute_task_min()  
        self.transform = transform


    def _read_labels(self,path):
        with open(path,"r+") as f:
            lines = f.readlines()
        color_pairs = ['black','blue','brown','gray','green','orange','pink','purple','red','white','yellow']
        labels = {
            'data': [],
            'upper_color':{
                'black': [],   
                'blue': [],
                'brown': [],
                'gray': [],
                'green': [],
                'orange': [],   
                'pink': [],
                'purple': [],
                'red': [],
                'white': [],
                'yellow': [],
            },
            'lower_color': {
                'black': [],
                'blue': [],
                'brown': [],
                'gray': [],
                'green': [],
                'orange': [],
                'pink': [],
                'purple': [],
                'red': [],
                'white': [],
                'yellow': []
            },
            'gender': {
                'male': [],
                'female': []
            },
            'bag': {
                'no': [],
                'yes': []
            },
            'hat': {
                'no': [],
                'yes': []
            }
        }

        for l in lines:
            l = l.replace("\n","")
            line = l.split(",")
            notes = []
            if self.target_label is None:
                # if "-1" not in line[1:]:
                notes.append(line[0])
                notes.append(int(line[1])-1)
                notes.append(int(line[2])-1)
                notes.append(int(line[3]))
                notes.append(int(line[4]))
                notes.append(int(line[5]))
            if self.target_label == 'all':
                if "-1" not in line[1:]:
                    notes.append(line[0])
                    notes.append(int(line[1])-1)
                    notes.append(int(line[2])-1)
                    notes.append(int(line[3]))
                    notes.append(int(line[4]))
                    notes.append(int(line[5]))
                    labels.append(notes)
            
            elif "-1" != line[self.target_label+1]:  # filters NA values1
                notes.append(line[0])
                notes.append(int(line[1])-1)
                notes.append(int(line[2])-1)
                notes.append(int(line[3]))
                notes.append(int(line[4]))
                notes.append(int(line[5]))
                color_uc = color_pairs[notes[1]]
                color_lc = color_pairs[notes[2]]
                gender = ('male' if notes[3] == 0 else 'female')
                bag_presence = ('no' if notes[4] == 0 else 'yes')
                hat_presence = ('no' if notes[5] == 0 else 'yes')
                
                labels['upper_color'][color_uc].append(len(labels['data']))
                labels['lower_color'][color_lc].append(len(labels['data']))
                labels['gender'][gender].append(len(labels['data']))
                labels['bag'][bag_presence].append(len(labels['data']))
                labels['hat'][hat_presence].append(len(labels['data']))
                labels['data'].append(notes)

        
        
            
                #labels.append(notes)
        return labels
    
    def _compute_task_min(self):
        task = self.task_name[self.target_label]
        data_idx_dict = self.labels[task]
        task_min = {}
        for k in data_idx_dict.keys():
            task_min.update({k:len(data_idx_dict[k])})
        return task_min

    def _preprocess_task_min(self):    # rispetto alla target_label
        min_val = self.task_min[min(self.task_min, key=self.task_min.get)]
        for k in self.task_min:
            self.task_min[k] = int(self.task_min[k]/min_val)  # int(self.task_min[k]/min_val)    
            
    def _shuffle_data(self):
        task = self.task_name[self.target_label]
        for k in self.labels[task].keys():
            random.shuffle(self.labels[task][k])
    
    def __iter__(self):
        task = self.task_name[self.target_label]
        self.task_min = self._compute_task_min()
        if self.shuffle:
            self._shuffle_data()
        iters = {}
        for k in self.task_min:
            iters.update({k:iter(self.labels[task][k])})
        #print(iters)
        min_label = min(self.task_min, key=self.task_min.get)
        min_val = self.task_min[min_label]
        self._preprocess_task_min()
        #print(self.task_min, min_label, min_val)
        # split load to workers
        # worker_info = torch.utils.data.get_worker_info()
        # start = 0
        # stop = min_val
        # if worker_info is None:
        #     iter_start = start
        #     iter_stop = stop
        # else:
        #     per_worker = int(math.ceil((stop - start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = start + worker_id * per_worker
        #     iter_stop = min(iter_start + per_worker, stop)
        for i in range(min_val):
            for k in self.task_min:
                for _ in range(self.task_min[k]):
                    try:
                        data_idx = next(iters[k])
                        img_path = self.labels['data'][data_idx][0]
                        img = Image.open(os.path.join(self.data_dir,img_path)).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        label = torch.tensor(self.labels['data'][data_idx][self.target_label+1])
                        yield img, label                        
                    except StopIteration:
                        continue
        return 
    def __len__(self):
        return len(self.labels['data'])

    # def __getitem__(self, idx):
    #     print('IDX', idx, type(self.labels['data'][idx]))
    #     img_path = self.labels['data'][idx][0]
    #     img = Image.open(os.path.join(self.data_dir,img_path)).convert('RGB')

    #     if self.transform:
    #         img = self.transform(img)
        
    #     label = torch.tensor(self.labels['data'][idx][self.target_label+1])


    #     return img, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((90,220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lbl = 0
    num_classes = 11
    save_path = 'models/best_model_b_alexnet_batch_mod_asym_V2.pth'

## b_path: 'models/best_model_b_alexnet_batch_mod_asym.pth'
    
    train_set = CustomDataset("training_set","training_set.txt", transform, target_label=lbl)
    val_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=lbl, shuffle=False)

    lossFun = nn.CrossEntropyLoss()
    from asym_loss import ASLSingleLabel
    lossFun = ASLSingleLabel(gamma_pos=torch.zeros(1,11).to('cuda'),gamma_neg=torch.tensor([1, 2, 3, 2, 3, 4, 4, 4, 3, 2, 4]).to('cuda'))
    #lossFun = ASLSingleLabel(gamma_pos=torch.zeros(1,2).to('cuda'),gamma_neg=torch.tensor([0, 3]).to('cuda'))

    model = SimpleModel(num_classes, 'alexnet')
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  ##### MODIFICARE
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", device)
    model.to(device)

    ckt = torch.load('models/best_model_uc_alexnet_batch_mod_asym_mod.pth')
    model.load_state_dict(ckt['model_state_dict'])

    val_loss_min = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_batch_size, val_batch_size = 32, 16
        val_loader = DataLoader(val_set, batch_size=val_batch_size)
        train_loader = DataLoader(train_set, batch_size=train_batch_size)
        model.train()
        running_loss = 0.0
        loss = 0
        correct = 0
        count_train = 0
        # for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = lossFun(outputs, labels)
        #     correct += ((outputs.argmax(dim=1)) == labels).sum().item()
        #     count_train += labels.shape[0]
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
        # train_acc = correct

        val_loss = 0
        count_val = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += ((outputs.argmax(dim=1)) == labels).sum().item()
                count_val += labels.shape[0]
                loss = lossFun(outputs, labels)
                val_loss += loss.item()

        # scheduler.step(val_loss)

        # if val_loss < val_loss_min:
        #     val_loss_min = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': val_loss,
        #         }, save_path)
        #     print('SAVED best model')
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter >= early_stopping_patience:
        #     print("Early stopping triggered.")
        #     break

        # print("TRAIN LOSS EPOCH:", epoch, ":", running_loss*train_batch_size/count_train,"ACC:",train_acc/count_train)
        print("VAL LOSS EPOCH:", epoch, ":", val_loss*val_batch_size/count_val,"ACC:",correct/count_val)
