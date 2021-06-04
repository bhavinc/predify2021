########################################
# Measure the correlation distances

# A brief explanation : 
# We pass clean images through the network and get the representations : say clean_e_n(t)
# Similarly we pass noisy images through the network and get the representations : say noisy_e_n(t)
# Now, given that the network does not the clean image (or neither has learnt it since it is from validation dataset), if noisy_e_n gets closer to clean_e_n, that that would be an emergent property of the dynamics.
# Hence, here we measure the correlation distance between clean_e_n and noisy_e_n for all timesteps t (0 =< t =< T) 
########################################



#%%

gpu_to_use       = '0'
number_of_images = 1000
batchsize        = 1
MAX_TIME_STEP    = 15
NOISES           = [0.25,0.5,0.75,1.,2.]
imagenet_root    = '/path/to/imagenet/dataset'



import os
from os.path import join as opj
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)


import csv
import numpy as np
import matplotlib.pyplot as plt
from   datetime import datetime
import pickle

import torch
import torchvision
from torchvision.datasets import ImageNet
import torchvision.models as models
from torchvision.transforms import transforms

device = torch.device('cuda:0')
rseed = 17
np.random.seed(rseed)
torch.manual_seed(rseed)


from ..model_factory import get_model


hps  = [
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[0]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[1]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[2]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[3]},
        {"ffm":0.8, "fbm":0.1,  "erm":0.01*lws[4]},
    ]


model = get_model('pvgg',pretrained=True,deep_graph=False,hyperparams=hps)



def corrdist(x,y):

    if len(x.shape) == len(y.shape) == 3:
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        return 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        

################################################
#       Dataset and train-test helpers
################################################
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



val_ds     = ImageNet(imagenet_root, split='val', download=False, transform=transform_val)
indices    = np.random.permutation(len(val_ds)).tolist()
val_ds     = torch.utils.data.Subset(val_ds, indices[:number_of_images])
val_loader = torch.utils.data.DataLoader(val_ds,  batch_size=batchsize, shuffle=True, drop_last=False,num_workers=16,pin_memory=True)




def get_corr_list(net,hp_t0,MAX_TIME_STEP):

    net.eval()
    net.reset()
    set_hyperparams(net,hp_t0)
    

    reps = {noise:None for noise in NOISES}
    recs = {noise:None for noise in NOISES}
    corr_dicts = {noise:None for noise in NOISES}

    for NOISE in NOISES:

        corr_list = np.zeros((MAX_TIME_STEP,net.number_of_pcoders))


        for i,(inputs,labels) in enumerate(val_loader,0):
                
            # get reps for clean images
            reps[NOISE] = []
            reps[0] = []
            for t in range(MAX_TIME_STEP):
                if t==0:
                    with torch.no_grad():
                        outputs = net(inputs.to(device))
                else:
                    with torch.no_grad():
                        outputs = net(None)            

                list_of_reps = [getattr(net,f"pcoder{i+1}").rep.detach().clone().cpu() for i in range(net.number_of_pcoders)]
                reps[0].append(list_of_reps)

                
            # get reps for noisy images
            net.reset()
            inputs = inputs.to(device) + torch.normal(0, NOISE, size=inputs.shape,generator=torch.manual_seed(0)).to(device)
            for t in range(MAX_TIME_STEP):
                if t==0:
                    with torch.no_grad():
                        outputs = net(inputs.to(device))
                else:
                    with torch.no_grad():
                        outputs = net(None)            

                list_of_reps = [getattr(net,f"pcoder{i+1}").rep.detach().clone().cpu() for i in range(net.number_of_pcoders)]
                reps[NOISE].append(list_of_reps)


            # calculate the correlation distances
            corr = np.zeros((MAX_TIME_STEP,net.number_of_pcoders))
            for t in range(MAX_TIME_STEP):
                for pcoder in range(net.number_of_pcoders):
                    corr[t,pcoder] = corrdist(reps[NOISE][t][pcoder][0],reps[0][t][pcoder][0])
            corr_list = np.dstack((corr_list,corr))

        corr_dicts[NOISE] = corr_list

    return corr_dicts




tstart = datetime.now()
corrects_noise_dict = get_corr_list(net,hp_t0,MAX_TIME_STEP)
tend = datetime.now()
print ("Time taken :",tend-tstart)


plt.style.use('default')
plt.figure(figsize=(12,8))
plt.suptitle(f"PVGG",fontsize=16)
for i,(noise,data) in enumerate(corrects_noise_dict.items()):

    plt.subplot(2,3,i+1)
    plt.title(f'Noise $\sigma$={noise}')
    for block_number in range(5):
        ydata = data[:,block_number,1:].mean(1)
        ydata = ydata/ydata[0]

        plt.plot(ydata,label=f"Pcoder {block_number+1}")
        plt.xlabel('Timesteps',fontsize=14)
        plt.ylabel('Normalized Correlation Distance',fontsize=14)
    plt.legend(bbox_to_anchor=(1,1))
    plt.xticks([1,15],[1,15],fontsize=14)
    plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('corr_dists_pvgg.pdf',bbox_inches='tight')
plt.show()

