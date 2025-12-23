from model import CSVDtransformer
from dataloader import load_Nifti_data_multimodal
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import nibabel as nib
import pandas as pd
import numpy as np
import argparse

def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-data_dir', default='./example_data/', help='Data Directory')
    return parser

parser = cli_parser()
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = str(args.data_dir)


####load csvd models
patch_size = 1024
embed_dim = 512
depth = 6
num_heads = 8
nclass_pred1 = [4, 4, 7, 4, 2, 3]
shift_index = np.load('./model_ckpt/csvd_shift_index_t2.npy')
shift_index1 = np.load('./model_ckpt/csvd_shift_index_t1.npy')

model = CSVDtransformer(input_size=[len(shift_index), len(shift_index)], patch_size=[patch_size, patch_size], in_chans = 1, out_chans=1, embed_dim=embed_dim, 
                                depth=depth, num_heads=num_heads, mlp_ratio=4.,qkv_bias=False, qk_scale=None, 
                                norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,
                                use_checkpoint=False, conv=True, skip=False, 
                                attn_drop=0,
                                proj_drop=0, 
                                pred_drop=0,
                                out_class=nclass_pred1,
                                cov_dim=0)
model.eval()
state_dic = torch.load('./model_ckpt/csvd_model.pth', 'cpu')
xx, yy = model.load_state_dict(state_dic, strict=False)
print(xx)
print(yy)
model.to(device)


####load cmb models
patch_size = 2048
embed_dim = 256
depth = 12
num_heads = 8
nclass_pred2 = [2, 3]
shift_index2 = np.load('./model_ckpt/csvd_shift_index_swi.npy')

model1 = CSVDtransformer(input_size=[len(shift_index2)], patch_size=[patch_size], in_chans = 1, out_chans=1, embed_dim=embed_dim, 
                                depth=depth, num_heads=num_heads, mlp_ratio=4.,qkv_bias=False, qk_scale=None, 
                                norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,
                                use_checkpoint=False, conv=True, skip=False, 
                                attn_drop=0,
                                proj_drop=0, 
                                pred_drop=0,
                                out_class=nclass_pred2,
                                cov_dim=0)
model1.eval()
state_dic = torch.load('./model_ckpt/cmb_model.pth', 'cpu')
xx, yy = model1.load_state_dict(state_dic, strict=False)
print(xx)
print(yy)
model1.to(device)


data_list_t1 = sorted(glob.glob(f'{data_dir}/*/T1_brain_1mm_stdspace.nii.gz'))
data_list_t2 = sorted(glob.glob(f'{data_dir}/*/T2_brain_1mm_stdspace.nii.gz'))
data_list_swi = sorted(glob.glob(f'{data_dir}/*/SWI_brain_1mm_stdspace.nii.gz'))
print('Data for analysis...')
print(data_list_t1)
print(data_list_t2)
print(data_list_swi)



##process data
import nibabel as nib
##read in mask
info = nib.load('./wm_mask_1mm.nii.gz')
image_mask = info.get_fdata()
image_mask = torch.FloatTensor(image_mask).unsqueeze(0).unsqueeze(0)        
image_mask[torch.isnan(image_mask)] = 0.0
image_mask[torch.isinf(image_mask)] = 0.0      
image_mask = torch.nn.functional.interpolate(image_mask, (182, 218, 182), mode='nearest')[0,0,:,:,:].numpy()
image_mask = image_mask>0.2
print(image_mask.shape)


##data loader
dataset = load_Nifti_data_multimodal(data_list_t1, data_list_t2, data_list_swi, image_mask)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
print(len(loader))

model.eval()
print('Inference begin...')
with torch.no_grad():
    
    risk_score = np.zeros((len(data_list_t1), 6))

    for batch_idx, _batch in enumerate(loader):
        # print(batch_idx)
        # print(_batch[0].shape)
        data_in_t2 = _batch[0].to(device)[:,:,shift_index]       
        data_in_t1 = _batch[1].to(device)[:,:,shift_index1]
        data_in_swi = _batch[2].to(device)[:,:,shift_index2]
        
        pred = model([data_in_t2, data_in_t1])

        pred1 = model1([data_in_swi])
        
        for ijk in range(0,5):
            prob = torch.nn.functional.softmax(pred[ijk].detach(), dim=1).cpu().numpy()
            risk_score[_batch[-1].cpu().numpy(),ijk] = np.sum(prob * np.expand_dims(np.arange(0, pred[ijk].shape[1]), 0), axis=1)

        prob = torch.nn.functional.softmax(pred1[0].detach(), dim=1).cpu().numpy()
        risk_score[_batch[-1].cpu().numpy(),-1] = np.sum(prob * np.expand_dims(np.arange(0, pred1[0].shape[1]), 0), axis=1)

print('Inference finished...')        


all_results1 = pd.DataFrame({'t1': data_list_t1, 't2': data_list_t2, 'swi': data_list_swi})
all_results2 = pd.DataFrame(risk_score, columns = ['PWMH','DWMH','Fezakasscore','EPVS','LI', 'CMB'])
all_results = pd.concat((all_results1, all_results2), axis=1)
all_results.to_csv('./Inference_results.csv')

