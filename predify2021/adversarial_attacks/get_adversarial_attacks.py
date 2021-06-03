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





model_name = sys.argv[3]
rseed = 17
gpu_to_use = sys.argv[1]
batchsize = 4
imagelist_fname = f'./{model_name}_0.8ff_0.1fb_0.01erm_1000images_rseed420'
TIMESTEP = int(sys.argv[2])
ckpt_freq = None





num_steps = 100
attack    = fb.attacks.LinfBIM(steps=num_steps,random_start=True,rel_stepsize=2.5/num_steps)
epsilons  = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
is_attack_targeted = True 


save_dir = 'attacks_tmp/'
save_file = f"{model_name}_timestep{TIMESTEP}_LinfBIM{num_steps}steps_{model_name}_0.8ff_0.1fb_0.01erm_rseed{rseed}_1000images.p"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
device = torch.device('cuda:0')
torch.manual_seed(rseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(rseed)
np.random.seed(rseed)



def get_dist(A,B):

    n,h,w,c = A.shape
    A = A.reshape(n,h*w*c)

    n,h,w,c = B.shape
    B = B.reshape(n,h*w*c)

    return (A - B).norm(p=float('inf'),dim=1)




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
successes = []      #successes[batch][epsilon][img_in_a_batch]

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
    # perturbs_for_this_batch = [(clipped_advs[eps].cuda() - images.cuda()).norm(dim=(1,2,3)) for eps in range(len(epsilons))]  # TODO: Change this if you are planning to change the attack
    perturbs_for_this_batch = [get_dist(clipped_advs[eps].to(device),images.to(device)) for eps in range(len(epsilons))]  # TODO: Change this if you are planning to change the attack
    perturbations.append(perturbs_for_this_batch)
    successes.append(success)


    if ckpt_freq is not None and ind+1%ckpt_freq == 0:
        tend = datetime.now()
        print (f"Checkpointing at index {ind}...")
        data_dict = {
                        'model':f"{model_name}_at_timestep{TIMESTEP}",
                        'number_of_images':1000,
                        'foolbox_version':fb.__version__,
                        'attack':attack,
                        'perturbations':perturbations,
                        'clipped_advs': clipped_images,
                        'successes':successes,
                        'targeted':is_attack_targeted,
                        'random_seed':rseed,
                        'epsilons':epsilons,
                    }

        with open(os.path.join(save_dir,f"ckpt_ind{ind}_{save_file}"),'wb') as f:
            pickle.dump(save_file,f)
        print (f'\nTime taken for CKPT {ind} : {tend-tstart}')


tend = datetime.now()
print ('\nTime taken : ',tend-tstart)



## Final save ##
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





























# corrects = [0.,0.,0.,0.]
# for ind,net in enumerate(model_list):
#     for i,(inputs,labels) in enumerate(val_loader,0):
#         net.reset()
#         inputs = inputs.to(device) + torch.normal(0, 0.5, size=inputs.shape,generator=torch.manual_seed(0)).to(device)
#         preds = net(inputs)            
#         outputs = preds.max(-1)[1]
#         corrects[ind] += torch.sum(outputs==labels.to(device)).cpu().clone()
       
#     print (x,'=====>',corrects[ind]/100.)

# print ()
# print ()
# print ('--'*20)
# print ('The accuracies are as follows : ')
# print ('--'*20)
# for i,t in enumerate([1,6,10]):
#     print (t,'------',corrects[i]/100.)

#%%
