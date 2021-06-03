#%%
######################################
##      Arguments
######################################
model_name = 'peffb0'
gpu_to_use = 0
rseed = 420
save_name = f'./{model_name}_0.8ff_0.1fb_0.01erm_1000images_rseed{rseed}'
batchsize  = 1


#####################################
##      Imports
#####################################

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import foolbox as fb;print (fb.__version__)
import torchvision
import random


from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
device = torch.device('cuda:0')
from os.path import join as opj


# Fix the seeds. Everything has to be deterministic here
torch.manual_seed(rseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(rseed)
np.random.seed(rseed)


os.mkdir(save_name)


#####################################
##      Load your CNN
#####################################
from ..model_factory import get_model


if model_name == 'pvgg':
    hps = [
            {"ffm":0.8, "fbm":0.1,  "erm":0.01},
            {"ffm":0.8, "fbm":0.1,  "erm":0.01},
            {"ffm":0.8, "fbm":0.1,  "erm":0.01},
            {"ffm":0.8, "fbm":0.1,  "erm":0.01},
            {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        ]
if model_name == 'peffb0':
    hps = [
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01},

    ]

model = get_model(model_name,pretrained=True,deep_graph=False,hyperparams=hps)
model.eval()
model.to(device)

#####################################
##       Create ImageList
#####################################

# This is the list of images that will be used for attacks.
# The constraint here is that it should be classified correctly across all timesteps of the cnn

def get_imagelist(cnn, dataset, timesteps, number_of_images=500,verbose=True):

    cnn.reset()
    cnn.eval()



    if dataset=='imagenet':

        from torchvision.datasets import ImageNet
        from torchvision.transforms import transforms

        transform_val = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),

        ])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        data_root  = '/home/milad/pcproject/imagenet'
        val_ds     = ImageNet(data_root, split='val', download=False, transform=transform_val)
        dataloader = torch.utils.data.DataLoader(val_ds,  batch_size=1, shuffle=True, drop_last=False,pin_memory=True)

    if verbose:
        print (f"Got the {dataset} dataloader with batchsize=1 and shuffle=True")


    if dataset=='cifar':
        raise NotImplementedError # we chose to use foolbox 2.4 for AA's paper



    imagelist = []
    for ind,(inputs,labels) in enumerate(dataloader):

        cnn.reset()

        for t in range(timesteps):

            if t ==0:
                with torch.no_grad():
                    outputs = cnn(normalize(inputs.to(device)))
            else:
                with torch.no_grad():
                    outputs = cnn(None)



            _,preds = outputs.max(-1)
            
            # get the flag at each timestep
            image_flag = preds == labels.to(device)

            #abort if False at any timestep
            if image_flag == False:
                print (f'Skipping Image {ind} with label {labels.item()} as it was classified as {preds.item()}')
                break
        
        #if it is still true, append it to the imagelist,otherwise continue the dataloader loop
        if image_flag == True:
            imagelist.append((inputs.cpu(),labels.cpu()))
            with open(opj(save_name,f'image{ind}.p'),'wb') as f:
                pickle.dump((inputs.cpu(),labels.cpu()),f)


            if verbose:
                print (f"Image {ind} with label {labels.item()} added")

        #abort if desired number of images reached        
        if len(imagelist) == number_of_images: 
            break

    return imagelist

tstart = datetime.now()
imagelist = get_imagelist(model,'imagenet',timesteps = 10, number_of_images = 1000)
tend = datetime.now()
print ("Total time taken for the imagelist :",tend-tstart)



#%%