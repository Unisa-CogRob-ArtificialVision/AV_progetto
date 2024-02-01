import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from par_train_custom import SimpleModel, CustomDataset
from tqdm import tqdm


class PARModel():
    def __init__(self, models_path, device, backbone=['vgg11']*5):

        self.device = device

        model_uc = SimpleModel(11,backbone[0])
        checkpoint = torch.load(models_path['uc_model'])
        model_uc.load_state_dict(checkpoint['model_state_dict'])
        model_uc.requires_grad = False

        model_lc = SimpleModel(11,backbone[1])
        checkpoint = torch.load(models_path['lc_model'])
        model_lc.load_state_dict(checkpoint['model_state_dict'])
        model_lc.requires_grad = False

        model_g = SimpleModel(2,backbone[2])
        checkpoint = torch.load(models_path['g_model'])
        model_g.load_state_dict(checkpoint['model_state_dict'])
        model_g.requires_grad = False

        model_b = SimpleModel(2,backbone[3])
        checkpoint = torch.load(models_path['b_model'])
        model_b.load_state_dict(checkpoint['model_state_dict'])
        model_b.requires_grad = False

        model_h = SimpleModel(2,backbone[4])
        checkpoint = torch.load(models_path['h_model'])
        model_h.load_state_dict(checkpoint['model_state_dict'])
        model_h.requires_grad = False

        self.model_uc = model_uc; self.model_uc.to(device); self.model_uc.eval()
        self.model_lc = model_lc; self.model_lc.to(device); self.model_lc.eval()
        self.model_g = model_g; self.model_g.to(device); self.model_g.eval()
        self.model_b = model_b; self.model_b.to(device); self.model_b.eval()
        self.model_h = model_h; self.model_h.to(device); self.model_h.eval()


    def predict(self,x):
        out_uc = self.model_uc(x)
        out_lc = self.model_lc(x)
        out_g = self.model_g(x)
        out_b = self.model_b(x)
        out_h = self.model_h(x)
        return [out_uc,out_lc,out_g,out_b,out_h]
    

# if __name__ == '__main__':

