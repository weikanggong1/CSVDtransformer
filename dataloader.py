from sklearn.preprocessing import quantile_transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from scipy.linalg import lstsq
import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class load_Nifti_data_multimodal(Dataset):
    def __init__(self, data_list_t1, data_list_t2, data_list_swi, img_mask):
        self.data_list_t1 = data_list_t1
        self.data_list_t2 = data_list_t2
        self.data_list_swi = data_list_swi
        self.img_mask = img_mask>0

    def __len__(self):
        return len(self.data_list_t1)

    def __getitem__(self, index):
        
        ###for T2 FLAIR
        ff = self.data_list_t2[index]
        image_x = nib.load(ff).get_fdata()
        image_x = torch.FloatTensor(image_x).unsqueeze(0).unsqueeze(0)        
        image_x[torch.isnan(image_x)] = 0.0
        image_x[torch.isinf(image_x)] = 0.0      
        image_x = torch.nn.functional.interpolate(image_x,(182, 218, 182), mode='nearest')[0,0,:,:,:]
        image_x = image_x.numpy()[self.img_mask]
        image_x1 = (image_x - image_x.mean())/image_x.std()       
        image_x1 = torch.FloatTensor(image_x1).unsqueeze(0)  

        ###for T1w
        ff = self.data_list_t1[index]
        image_x = nib.load(ff).get_fdata()
        image_x = torch.FloatTensor(image_x).unsqueeze(0).unsqueeze(0)        
        image_x[torch.isnan(image_x)] = 0.0
        image_x[torch.isinf(image_x)] = 0.0      
        image_x = torch.nn.functional.interpolate(image_x,(182, 218, 182), mode='nearest')[0,0,:,:,:]
        image_x = image_x.numpy()[self.img_mask]
        image_x2 = (image_x - image_x.mean())/image_x.std()        
        image_x2 = torch.FloatTensor(image_x2).unsqueeze(0)      


        ###for SWI
        ff = self.data_list_swi[index]
        image_x = nib.load(ff).get_fdata()
        image_x = torch.FloatTensor(image_x).unsqueeze(0).unsqueeze(0)        
        image_x[torch.isnan(image_x)] = 0.0
        image_x[torch.isinf(image_x)] = 0.0      
        image_x = torch.nn.functional.interpolate(image_x,(182, 218, 182), mode='nearest')[0,0,:,:,:]
        image_x = image_x.numpy()[self.img_mask]
        image_x3 = quantile_transform(image_x.reshape(-1, 1), n_quantiles=1000, random_state=0, copy=True, output_distribution='normal').flatten()
        image_x3 = torch.FloatTensor(image_x3).unsqueeze(0)      

    
        return image_x1, image_x2, image_x3, index
