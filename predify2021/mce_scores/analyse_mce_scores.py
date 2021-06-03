########################################
##   Analyse the mCE scores
########################################
import numpy as np
import matplotlib.pyplot as plt
from   datetime import datetime
import pickle

fname = ''

total_images  = 50000.


plt.style.use('default')
fp = {'fontsize':16}

import scipy
import scipy.stats

def mean_confidence_interval(data,confidence=0.95):

    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    std = np.std(a)
    h = std*scipy.stats.t.ppf((1 + confidence)/2., n-1)
    return m,std,h


with open(fname,'rb') as f:
    accuracy_dict = pickle.load(f)    ## accuracy_dict[timesteps] = {distortions : [5  accuracies ] }


for t,v in accuracy_dict.items():
    for k1,v1 in v.items():
        v[k1] =  np.array([1. - (x/total_images) for x in v1]).sum()
    accuracy_dict[t] = v



dict_of_means = {}

ydata = []
for noise,avg_err in accuracy_dict.items():
    data = np.array(list(avg_err.values()))

    ### bootstrap
    list_of_means = []    
    for _ in range(1000):

        x = np.random.choice (data,size=100000,replace=True)
        list_of_means.append(np.mean(x))     

    dict_of_means[noise] = list_of_means


#%%
###get the datapoints and confidence intervals

confidence_intervals = []
ydata = []
for _time in accuracy_dict.keys():
    mean,error,_ = mean_confidence_interval(dict_of_means[_time])
    ydata.append(mean)
    confidence_intervals.append(error)


###plot 1 
xdata = np.arange(11)

plt.figure(figsize=(5,5))
plt.subplot(1,1,1)
plt.title('PEfficientNetB0',**fp)
plt.plot(xdata,ydata/ydata[0],color="#1f77b4")
plt.errorbar(xdata,ydata/ydata[0],yerr=confidence_intervals[1]/ydata[0])
plt.axhline(ydata[0]/ydata[0],linestyle='dashed',color="#b30006")

plt.xlabel('Timesteps',**fp)
plt.ylabel('mCE scores',**fp)




plt.tight_layout()
plt.savefig(f'mce.pdf',bbox_inches='tight')

#%%




















#%%










import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.interpolate import make_interp_spline,BSpline

np.random.seed(17)


def get_bootstrap_estimates(data,number_of_samples_to_draw,num_bootstraps=1000,operation='mean'):

    import scipy
    from scipy import stats
    def mean_confidence_interval(data,confidence=0.95):

        a = 1.0 * np.array(data)
        n = len(a)
        m = np.mean(a)
        std = np.std(a)
        # se = scipy.stats.sem(a)
        h = std*scipy.stats.t.ppf((1 + confidence)/2., n-1)
        # return  m, m-h,m+h,h
        return m,std,h



    samples = []
    for _ in range(num_bootstraps):
        x = np.random.choice(data,size=number_of_samples_to_draw,replace=True)
        if operation=='mean':
            samples.append(np.mean(x))
        elif operation=='median':
            samples.append(np.median(x))


    mean,std,conf_interval = mean_confidence_interval(samples,confidence=0.95)

    return  mean,std,conf_interval


all_distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'frost', 'fog', 'brightness', 'snow',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


def plot2(fname):
    accuracy_dict = pickle.load(open(fname,'rb'))

    # accuracy_dict[timesteps] = {distortions : [5  accuracies ] }



    fp = {'fontsize':16}

    plt.figure(figsize=(12,16))
    plt.suptitle(f'CE scores : {tag}',**fp)
    counter = 1
    for distortions in all_distortions:

        bootstrap_estimates = {t:{} for t in accuracy_dict.keys() }

        for time,values in accuracy_dict.items():

            for x, corrects in enumerate(values[distortions]):

                corrects = int(corrects)

                population_sample = np.zeros(50000)
                population_sample[:corrects] = 1
                np.random.shuffle(population_sample)

                assert np.sum(population_sample) == corrects


                if x ==0:
                    new_population_data = population_sample
                else:
                    new_population_data = np.append(new_population_data,population_sample)

            assert len(new_population_data) == 5*50000

            mean,std,confs = get_bootstrap_estimates(population_sample,5000,num_bootstraps=2000)
            bootstrap_estimates[time] = 1. - mean,std,confs

        plt.subplot(5,4,counter)
        factor = 1./bootstrap_estimates[0][0]
        plt.errorbar(x = [0,1,2,3,4,5,6,7,8,9,10],y = [bootstrap_estimates[time][0]*factor for time in range(11)] ,yerr=[bootstrap_estimates[time][1]*factor for time in range(11)],color="#1f77b4")
        plt.axhline(1.,linestyle='dashed',color="#b30006")
        plt.title(distortions,**fp)
        plt.xlabel('Timesteps',fontsize=12)
        plt.xticks([0,10],[0,10],**fp)
        plt.yticks([0.96,1.02],[0.96,1.02],**fp)
        plt.ylabel('CE score')
        counter += 1
    plt.tight_layout()
    plt.savefig(f'{tag}_cescores.png')

plot2(fname)
