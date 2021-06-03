gpu_to_use =  '0'
batchsize=150
MAX_TIME_STEPS = 10


rseed = 17


############################

import os
from os.path import join as opj
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)


import csv
import numpy as np
import matplotlib.pyplot as plt
from   datetime import datetime
import pickle
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.transforms import transforms

from ..model_factory import get_model

device = torch.device('cuda:0')
np.random.seed(rseed)
torch.manual_seed(rseed)


transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])




hps = [
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[0]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[1]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[2]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[3]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[4]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[5]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[6]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[7]},

    ]


net = get_model('peffb0',pretrained=True,deep_graph=False,hyperparams=hps)



def eval_training(_net, dataloader, timesteps,tqdm_desc=''):

    
    corrects   = np.zeros((timesteps+1,1))
    for (images, labels) in tqdm(dataloader,desc=tqdm_desc):
        _net.reset()


        for tt in range(timesteps+1):
            if tt == 0:
                with torch.no_grad():
                    outputs = _net(images.to(device))
            else:
                with torch.no_grad():
                    outputs = _net(None)
        
            _, preds = outputs.max(-1)



            corrects[tt,0] += torch.sum(preds==labels.to(device))


    for tt in range(timesteps+1):
        acc = 100.*corrects[tt,0] / (len(dataloader.dataset))
        print(f'Test set t = {tt:02d}: Accuracy: {acc:.4f}')
    print()
    return corrects



def get_imagenetc_dataloader( noise_level,noise_type):
    
    
    imagenetc_dir = ''

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])


    distorted_dataset = ImageFolder( root= imagenetc_dir + noise_type + '/' + str(noise_level), transform= transform_val)
    distorted_dataset_loader = torch.utils.data.DataLoader(distorted_dataset, batch_size=batchsize, shuffle=False, num_workers=16,pin_memory=True)

    return distorted_dataset_loader




def main():



    all_noises = ["brightness",
                    "contrast",
                    "defocus_blur",
                    "elastic_transform",
                    "fog",
                    "frost",
                    "gaussian_blur",
                    "gaussian_noise",
                    "glass_blur",
                    "impulse_noise",
                    "jpeg_compression",
                    "motion_blur",
                    "pixelate",
                    "saturate",
                    "shot_noise",
                    "snow",
                    "spatter",
                    "speckle_noise",
                    "zoom_blur"]




    noise_levels=[1,2,3,4,5]

    accuracy_dict = {timestep:{noise:[0. for severity in range(len(noise_levels))] for noise in all_noises} for timestep in range(MAX_TIME_STEPS+1)}

    for noise in all_noises : 


        for noise_level in noise_levels:

            imagenetc_trainloader = get_imagenetc_dataloader( noise_level=noise_level,noise_type=noise)
            
            print ()
            print ('-'*30)
            print(f"TESTING THE NETWORK ON : NOISE {noise} SEVERITY {noise_level}")
            print ('-'*30)
            tstart = datetime.now()
            print (f"STARTING AT : {tstart}")
            accuracy = eval_training(net, imagenetc_trainloader, timesteps=MAX_TIME_STEPS,tqdm_desc=f"{noise}_{noise_level}")
            tend = datetime.now()
            print (f"TOTAL TIME TAKEN : {tend-tstart}")
            

            
            for t in range(MAX_TIME_STEPS+1):
                accuracy_dict[t][noise][noise_level - 1] = accuracy[t,0]

    with open(f"mce_peffb0.p",'wb') as f:
        pickle.dump(accuracy_dict,f)


main()

