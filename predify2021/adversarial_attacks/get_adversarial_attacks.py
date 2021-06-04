###########################################
# Get the adversarial attacks 
# using the imagelists.
###########################################
#%%
import sys

import os
from os.path import join as opj


import random 
import numpy as np
import matplotlib.pyplot as plt
from   datetime import datetime

import torch
import torchvision.models as models
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms


import pickle
from tqdm import tqdm
import foolbox as fb



gpu_to_use      = str(sys.argv[1])
TIMESTEP        = int(sys.argv[2])
model_name      = str(sys.argv[3])
imagelist_fname = str(sys.argv[4])


rseed = 17
batchsize = 4
ckpt_freq = None     #if your attacks take too long and you would like to checkpoint them 





num_steps = 100
attack    = fb.attacks.LinfBIM(steps=num_steps,random_start=True,rel_stepsize=2.5/num_steps)
epsilons  = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
is_attack_targeted = True 


save_dir = 'attacks_tmp/'
save_file = f"{model_name}_timestep{TIMESTEP}_LinfBIM{num_steps}steps_{model_name}_0.8ff_0.1fb_0.01erm_rseed{rseed}_1000images.p"


# setup devices and seeds
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
device = torch.device('cuda:0')
torch.manual_seed(rseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(rseed)
np.random.seed(rseed)



def get_dist(A,B,p_val=float('inf')):

    n,h,w,c = A.shape
    A = A.reshape(n,h*w*c)

    n,h,w,c = B.shape
    B = B.reshape(n,h*w*c)

    return (A - B).norm(p=p_val,dim=1)



# setup the dataloaders for the imagelist
class CustomDataSet(torch.utils.data.Dataset):

    def __init__(self, maindir, transform=None):
        self.maindir = maindir
        self.all_imgs = os.listdir(maindir)
        self.transform = transform

    def __getitem__(self, idx):
        image,label = pickle.load(open(opj(self.maindir,self.all_imgs[idx]),'rb')) 
        return image,label

    def __len__(self):
        return len(self.all_imgs)

val_ds = CustomDataSet(imagelist_fname)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=batchsize,shuffle=False,drop_last=False)


###########################################################
##                      Network    
###########################################################

from ..model_factory import get_model,set_hyperparams 

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

# Make sure that deep_graph=True here. Otherwise, torch will not create the deeper graphs
model = get_model(model_name,pretrained=True,deep_graph=True,timesteps=TIMESTEP,hyperparams=hps)


model.eval()
model.to(device)

#####################################
##          Attacks
#####################################


preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

fmodel = fb.PyTorchModel(model,bounds=(0,1),preprocessing=preprocessing)

perturbations = []  # perturbations[batch][epsilon][img_in_a_batch]
clipped_images = [] # clipped_images[batch][epsilon][whole adv batch (i.e. in NCHW format)]
successes = []      # successes[batch][epsilon][img_in_a_batch]

tstart = datetime.now()
for ind,(images,labels) in enumerate(tqdm(val_loader,ncols=50)):

    images = images.squeeze_(1)
    labels = labels.flatten()

    ### change the target classes
    if is_attack_targeted:
        target_class = (labels+800)%1000
        criterion = fb.criteria.TargetedMisclassification(target_class.to(device))
    else:
        criterion = fb.criteria.Misclassification(labels.to(device))
    
    raw_advs,clipped_advs,success = attack(fmodel,images.to(device),criterion=criterion,epsilons=epsilons)
    
    clipped_images.append(clipped_advs)
    perturbs_for_this_batch = [get_dist(clipped_advs[eps].to(device),images.to(device)) for eps in range(len(epsilons))]
    perturbations.append(perturbs_for_this_batch)
    successes.append(success)


    if ckpt_freq is not None and ind+1%ckpt_freq == 0:
        tend = datetime.now()
        print (f"Checkpointing at index {ind}...")
        data_dict = {
                        'model'           : f"{model_name}_at_timestep{TIMESTEP}",
                        'number_of_images': len(val_ds),
                        'foolbox_version' : fb.__version__,
                        'attack'          : attack,
                        'perturbations'   : perturbations,
                        'clipped_advs'    : clipped_images,
                        'successes'       : successes,
                        'targeted'        : str(is_attack_targeted),
                        'random_seed'     : rseed,
                        'epsilons'        : epsilons,
                        'imagelist_used'  : imagelist_fname
                    }

        with open(os.path.join(save_dir,f"ckpt_ind{ind}_{save_file}"),'wb') as f:
            pickle.dump(save_file,f)
        print (f'\nTime taken for CKPT {ind} : {tend-tstart}')


tend = datetime.now()
print ('\nTime taken : ',tend-tstart)



# final save
data_dict = {
    'model'           : f"{model_name}_at_timestep{TIMESTEP}",
    'number_of_images': len(val_ds),
    'foolbox_version' : fb.__version__,
    'attack'          : attack,
    'perturbations'   : perturbations,
    'clipped_advs'    : clipped_images,
    'successes'       : successes,
    'targeted'        : str(is_attack_targeted),
    'random_seed'     : rseed,
    'epsilons'        : epsilons,
    'imagelist_used'  : imagelist_fname
}

with open(os.path.join(save_dir,save_file),'wb') as f:
    pickle.dump(data_dict,f)

print ('Done.')

#%%
