import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as M
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import os
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from random import randint
from torchvision.transforms import v2
    
# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None, target_label=None, shuffle=True):
        self.data_dir = data_path
        self.target_label = target_label        # indica la label su cui voglio fare l'addestramento. Serve per scartare solo i dati che hanno "-1" per la label target
        self.task_name = ['upper_color', 'lower_color', 'gender', 'bag', 'hat']

        self.shuffle = shuffle
        self.labels = self._read_labels(labels_path)  
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
                if int(line[1]) == -1:
                    line[1] = str(int(line[1])+1)
                notes.append(int(line[1])-1)
                if int(line[2]) == -1:
                    line[2] = str(int(line[2])+1)
                notes.append(int(line[2])-1)
                notes.append(int(line[3]))
                notes.append(int(line[4]))
                notes.append(int(line[5]))

                if (notes[1] != -1):
                    color_uc = color_pairs[notes[1]]
                    labels['upper_color'][color_uc].append(len(labels['data']))
                if (notes[2] != -1):
                    color_lc = color_pairs[notes[2]]
                    labels['lower_color'][color_lc].append(len(labels['data']))
                if (notes[3] != -1):
                    gender = ('male' if notes[3] == 0 else 'female')
                    labels['gender'][gender].append(len(labels['data']))
                if (notes[4] != -1):
                    bag_presence = ('no' if notes[4] == 0 else 'yes')
                    labels['bag'][bag_presence].append(len(labels['data']))
                if (notes[5] != -1):
                    hat_presence = ('no' if notes[5] == 0 else 'yes')
                    labels['hat'][hat_presence].append(len(labels['data']))
                labels['data'].append(notes)

            if self.target_label == 'all':
                if "-1" not in line[1:]:
                    notes.append(line[0])
                    notes.append(int(line[1])-1)
                    notes.append(int(line[2])-1)
                    notes.append(int(line[3]))
                    notes.append(int(line[4]))
                    notes.append(int(line[5]))
                    labels["data"].append(notes)
        return labels
    
    def _shuffle_data(self):
        for task in self.task_name:
            for k in self.labels[task].keys():
                random.shuffle(self.labels[task][k])

    def __len__(self):
        return len(self.labels['data'])
    
    def __getitem__(self, idx):
       # print('IDX', idx, type(self.labels['data'][idx]))
        img_path = self.labels['data'][idx][0]
        img = Image.open(os.path.join(self.data_dir,img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels["data"][idx][1:])


class BalancedObjectsSampler(torch.utils.data.BatchSampler):
    """
    Tries to sample uniformly from every class: 
    fails miserably, but at least we get more than one sample per batch for every class
    """

    def __init__(self, dataset: CustomDataset):
        self.dataset = dataset
        self.batch_size = 28*4
        self.batch_count = math.ceil(len(self.dataset) / self.batch_size)
        
    def __iter__(self):
        tasks = ["upper_color", "lower_color", "gender", "bag", "hat"]
        for j in range(0,self.batch_count):
            indexes = []
            for i in range(0,4):
                for task in tasks:
                    # one for every task
                    for key in self.dataset.labels[task].keys():
                        # queue mode
                        single_index = self.dataset.labels[task][key].pop(0)
                        # check if same index is running or not
                        for task_ in tasks:
                            for key_ in self.dataset.labels[task_].keys():
                                if self.dataset.labels[task_][key_][0] == single_index:
                                    self.dataset.labels[task_][key_].pop()
                                    self.dataset.labels[task_][key_].append(single_index)
                        self.dataset.labels[task][key].append(single_index)
                        indexes.append(single_index)
            yield indexes # returns the custom batch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        #v2.RandomCrop(214),  # prova 
        #transforms.Resize((224,224))
        #transforms.Resize((90,220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip()
    ])

    lbl = 2
    num_classes = 2
    save_path = 'models/best_cbam_net.pth'

    train_set = CustomDataset("training_set","training_set.txt", transform, target_label=None)
    val_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=None, shuffle=False)

    from asym_loss import ASLSingleLabel

    loss_upper_color = ASLSingleLabel(gamma_pos=torch.zeros(1,11).to(device),gamma_neg=torch.tensor([1, 2, 3, 2, 3, 4, 4, 4, 3, 2, 4]).to(device)).to(device)
    loss_lower_color = ASLSingleLabel(gamma_pos=torch.zeros(1,11).to(device),gamma_neg=torch.tensor([1, 2, 3, 2, 3, 4, 4, 4, 3, 2, 4]).to(device)).to(device)
    loss_gender = ASLSingleLabel(gamma_pos=torch.zeros(1,2).to(device), gamma_neg=torch.tensor([3,1]).to(device)).to(device)
    loss_bag = ASLSingleLabel(gamma_pos=torch.zeros(1,2).to(device), gamma_neg=torch.tensor([5,1]).to(device)).to(device)
    loss_hat = ASLSingleLabel(gamma_pos=torch.zeros(1,2).to(device), gamma_neg=torch.tensor([6,1]).to(device)).to(device)

    from CBAM_net import CBAMnet
    from prova_cbam import Alexnet
    from vgg19_cbam import VGG19

    model = CBAMnet()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  ##### MODIFICARE
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    num_epochs = 100
    print("DEVICE:", device)
    model.to(device)

    val_loss_min = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    batch_sampler_train = BalancedObjectsSampler(train_set)
    batch_sampler_val = BalancedObjectsSampler(val_set)
    val_loader = DataLoader(val_set, batch_sampler=batch_sampler_val,num_workers=4)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train,num_workers=4)

    for epoch in range(num_epochs):
        train_batch_size, val_batch_size = 112, 112
        model.train()
        running_loss = 0.0
        loss = 0
        correct_uc = 0
        correct_lc = 0
        correct_g = 0
        correct_b = 0
        correct_h = 0

        count_train_uc = 0
        count_train_lc = 0
        count_train_g = 0
        count_train_b = 0
        count_train_h = 0

        count = 0

        train_set._shuffle_data()

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_upper_color, labels_upper_color = outputs["upper_color"], labels[:,0]
            outputs_lower_color, labels_lower_color = outputs["lower_color"], labels[:,1]
            outputs_gender, labels_gender = outputs["gender"], labels[:,2]
            outputs_bag, labels_bag = outputs["bag"], labels[:,3]
            outputs_hat, labels_hat = outputs["hat"], labels[:,4]
    
            loss_uc = loss_upper_color(outputs_upper_color, labels_upper_color)
            loss_lc = loss_lower_color(outputs_lower_color, labels_lower_color)
            loss_g = loss_gender(outputs_gender, labels_gender)
            loss_b = loss_bag(outputs_bag, labels_bag)
            loss_h = loss_hat(outputs_hat, labels_hat)

            correct_uc += (outputs_upper_color.argmax(dim=1) == labels_upper_color).sum().item()
            correct_lc += (outputs_lower_color.argmax(dim=1) == labels_lower_color).sum().item()
            correct_g += (outputs_gender.argmax(dim=1) == labels_gender).sum().item()
            correct_b += (outputs_bag.argmax(dim=1) == labels_bag).sum().item()
            correct_h += (outputs_hat.argmax(dim=1) == labels_hat).sum().item()

            count_train_uc += (labels_upper_color != -1).sum()
            count_train_lc += (labels_lower_color != -1).sum()
            count_train_g += (labels_gender != -1).sum()
            count_train_b += (labels_bag != -1).sum()
            count_train_h += (labels_hat != -1).sum()

            count += labels.shape[0]

            # simple loss aggregation
            loss = loss_uc + loss_lc + loss_g + loss_b + loss_h
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        train_acc_uc = correct_uc
        train_acc_lc = correct_lc
        train_acc_g = correct_g
        train_acc_b = correct_b
        train_acc_h = correct_h

        val_loss = 0

        correct_val_uc = 0
        correct_val_lc = 0
        correct_val_g = 0
        correct_val_b = 0
        correct_val_h = 0

        count_val_uc = 0
        count_val_lc = 0
        count_val_g = 0
        count_val_b = 0
        count_val_h = 0

        count_val = 0

        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                outputs_upper_color, labels_upper_color = outputs["upper_color"], labels[:,0]
                outputs_lower_color, labels_lower_color = outputs["lower_color"], labels[:,1]
                outputs_gender, labels_gender = outputs["gender"], labels[:,2]
                outputs_bag, labels_bag = outputs["bag"], labels[:,3]
                outputs_hat, labels_hat = outputs["hat"], labels[:,4]

                correct_val_uc += (outputs_upper_color.argmax(dim=1) == labels_upper_color).sum().item()
                correct_val_lc += (outputs_lower_color.argmax(dim=1) == labels_lower_color).sum().item()
                correct_val_g += (outputs_gender.argmax(dim=1) == labels_gender).sum().item()
                correct_val_b += (outputs_bag.argmax(dim=1) == labels_bag).sum().item()
                correct_val_h += (outputs_hat.argmax(dim=1) == labels_hat).sum().item()

                count_val += labels.shape[0]

                count_val_uc += (labels_upper_color != -1).sum()
                count_val_lc += (labels_lower_color != -1).sum()
                count_val_g += (labels_gender != -1).sum()
                count_val_b += (labels_bag != -1).sum()
                count_val_h += (labels_hat != -1).sum()

                loss_uc = loss_upper_color(outputs_upper_color, labels_upper_color)
                loss_lc = loss_lower_color(outputs_lower_color, labels_lower_color)
                loss_g = loss_gender(outputs_gender, labels_gender)
                loss_b = loss_bag(outputs_bag, labels_bag)
                loss_h = loss_hat(outputs_hat, labels_hat)
                
                loss = loss_uc + loss_lc + loss_g + loss_b + loss_h
                val_loss += loss.item()

        scheduler.step(val_loss)

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, save_path)
            print('SAVED best model')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        print("\n===============\nTRAIN LOSS EPOCH:\n", epoch, "--", running_loss*train_batch_size/count, "===================") 
        print("UPPER COLOR ACC: ",train_acc_uc/count_train_uc.item())
        print("LOWER COLOR ACC: ",train_acc_lc/count_train_lc.item())
        print("GENDER ACC: ",train_acc_g/count_train_g.item())
        print("BAG ACC: ",train_acc_b/count_train_b.item())
        print("HAT ACC: ",train_acc_h/count_train_h.item())
        print("\n================\nVAL LOSS EPOCH:\n", epoch, ":", val_loss*val_batch_size/count_val)
        print("UPPER COLOR ACC: ",correct_val_uc/count_val_uc.item())
        print("LOWER COLOR ACC: ",correct_val_lc/count_val_lc.item())
        print("GENDER ACC: ",correct_val_g/count_val_g.item())
        print("BAG ACC: ",correct_val_b/count_val_b.item())
        print("HAT ACC: ",correct_val_h/count_val_h.item())
        print("#########################################\n")

        # VAL LOSS EPOCH:
        #  3 : 3.4295847765896297
        # UPPER COLOR ACC:  0.588548492791612
        # LOWER COLOR ACC:  0.5535714285714286
        # GENDER ACC:  0.8186435124508519
        # BAG ACC:  0.7785878112712975
        # HAT ACC:  0.9407765399737876