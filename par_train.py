import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as M
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg11
import os
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0, alexnet, vgg11, convnext_large, ConvNeXt_Large_Weights, ResNet18_Weights, VGG11_Weights

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
        for param in self.backbone.parameters():
            param.requires_grad = False
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
class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None, target_label=None):
        self.data_dir = data_path
        self.target_label = target_label        # indica la label su cui voglio fare l'addestramento. Serve per scartare solo i dati che hanno "-1" per la label target

        self.labels = self._read_labels(labels_path)    
        self.transform = transform
        

    def _read_labels(self,path):
        with open(path,"r+") as f:
            lines = f.readlines()
        labels = []
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
                labels.append(notes)
            elif "-1" != line[self.target_label+1]:  # filters NA values1
                notes.append(line[0])
                notes.append(int(line[1])-1)
                notes.append(int(line[2])-1)
                notes.append(int(line[3]))
                notes.append(int(line[4]))
                notes.append(int(line[5]))
                labels.append(notes)
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.labels[idx][0]
        img = Image.open(os.path.join(self.data_dir,img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx][1:])


        return img, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((90,220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #############   
    lbl = 1             # 0 = upper_color; 1 = lower_color; 2 = gender; 3 = bag; 4 = hat;
    num_classes = 11    # 11 per upper_color e lower_color, 2 per gli altri
    save_path = 'models/best_model_lc_alexnet.pth'
    #############

    train_set = CustomDataset("training_set","training_set.txt", transform, target_label=lbl)
    val_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=lbl)
    
    #test_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=lbl)
    train_set, val_set =  torch.utils.data.random_split(train_set,[0.75,0.25])
    #val_set, _ =  torch.utils.data.random_split(val_set,[0.5,0.5])
    # print(len(train_set))


    #print(train_set[0],"\n",val_set[0])
    train_batch_size, val_batch_size = 32, 32
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

    lossFun = nn.CrossEntropyLoss()


    model = SimpleModel(num_classes,'alexnet')
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    ### training
    # checkpoint = torch.load('models/best_model_uc.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.requires_grad = True

    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:",device)
    model.to(device)

    val_loss_min = float('inf')



    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loss = 0
        correct = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels[:,lbl].to(device)  # prendo solo le label associate al colore dei 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFun(outputs, labels)
            correct += ((outputs.argmax(dim=1)) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = correct

        val_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels[:,lbl].to(device)  # prendo solo le label associate al colore dei vestiti
                outputs = model(inputs)
                correct += ((outputs.argmax(dim=1)) == labels).sum().item()
                loss = lossFun(outputs, labels)
                val_loss += loss.item()
        
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, save_path)
            print('SAVED best model')
        print("TRAIN LOSS EPOCH:", epoch, ":", running_loss/len(train_loader),"ACC:",train_acc/len(train_set))
        print("VAL LOSS EPOCH:", epoch, ":", val_loss/len(val_loader),"ACC:",correct/len(val_set))

    # print('TESTING')
    # ckt = torch.load(save_path)
    # model.load_state_dict(ckt['model_state_dict'])
    # with torch.no_grad():
        
    #     for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
    #         # if binary:
    #         #     inputs, labels = inputs.to(device), labels[:,2].to(device)  # prendo solo le label associate al colore dei vestiti
    #         # else:
    #         inputs, labels = inputs.to(device), labels[:,lbl].to(device)  # prendo solo le label associate al colore dei vestiti
    #         outputs = model(inputs)
    #         # if binary:
    #         #     outputs = act(outputs)
    #         # if binary:
    #         #     loss = lossFun(outputs, labels.unsqueeze(1).to(torch.float32))
    #         #     correct += ((outputs.squeeze() > 0.5) == labels).sum().item()
    #         # else:
    #         correct += ((outputs.argmax(dim=1)) == labels).sum().item()
    #         loss = lossFun(outputs, labels)
    #         val_loss += loss.item()
    #         #print(outputs,(outputs.argmax(dim=1)))
