

#%%

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy
from scipy import stats
from scipy.interpolate import make_interp_spline,BSpline

np.random.seed(17)


def convert_to_successes(A):
    B = torch.stack(A,dim=2)
    x,y,z = B.shape
    B = B.reshape(x,y*z)
    return B.sum(-1)


TIMESTEPS = [0,2,4,6,8]




NET = 'peffb0'



fp = {'fontsize':14}
plt.style.use('default')
plt.figure(figsize=(7,7))


ATTACKS = ['LinfBIM20steps']



for ATTACK in ATTACKS:

    perturbation_dict = {}
    for t in TIMESTEPS:
        fname = f"filename_here"
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                data_dict = pickle.load(f)
            assert NET in data_dict['model']

            print (data_dict['attack'])
            print (t,' : ',ATTACK,' : ',data_dict['epsilons'])
            print ()
            print ()
            perturbation_dict[t] = (data_dict['epsilons'],convert_to_successes(data_dict['successes']))


            del data_dict





    plt.subplot(1,1,counter)
    plt.title(f' {NET} : {ATTACK}',**fp)
    factor = 1.
    for time in perturbation_dict.keys():
        used_epsilons,successes = perturbation_dict[time]
        plt.plot(np.array(used_epsilons),successes.cpu().numpy()*factor,label=time,alpha=0.4,marker='.')
        plt.xscale('log')

    plt.xlabel('epsilons',**fp)
    plt.ylabel('Number of images where attack was successful',**fp)
    plt.legend(title='Timesteps',loc='lower right',**fp,title_fontsize=14)
plt.tight_layout()
plt.show()
#%%


chosen_epsilons = [0.7,0.8,0.9,1.0]
fp = {'fontsize':18}
plt.figure(figsize=(5,5))
plt.title(f' PEfficientNetB0 : {ATTACK}',**fp)

for chosen_epsilon in chosen_epsilons:
    ys = []
    xs = []
    for time in perturbation_dict.keys():
        used_epsilons,successes = perturbation_dict[time]
        xs.append(time)
        factor = 0.001
        ys.append(successes.cpu().numpy()[used_epsilons.index(chosen_epsilon)]*factor)
    plt.plot(xs,ys/ys[0],marker='.',label=chosen_epsilon)
plt.axhline(1.0,color='orange',linestyle='dashed')
plt.xlabel('Timesteps',**fp)
plt.ylabel('Success rate of the attack',**fp)
plt.legend()
plt.show()
#%%