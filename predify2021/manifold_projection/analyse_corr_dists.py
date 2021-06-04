# Just plotting the correlation data better

import numpy as np
import matplotlib.pyplot as plt
import torch


plt.style.use('default')
plt.figure(figsize=(12,8))

main_fname = f"/path/to/pickled/dicts"  
with open(main_fname,'rb') as f:
    corrects_noise_dict = pickle.load(f)

fp = {'fontsize':14}
for i,(noise,data) in enumerate(corrects_noise_dict.items()):

    plt.subplot(2,3,i+1)
    plt.title(f'Noise $\sigma$={noise}',**fp)
    for block_number in range(8):
        ydata = data[:,block_number,1:].mean(1)
        ydata = ydata/ydata[0]

        plt.plot(ydata,label=f"Pcoder {block_number+1}")
        plt.xlabel('Timesteps',**fp)
        plt.ylabel('Normalized Correlation Distance',**fp)
plt.xticks(np.arange(1,15),fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1,1),**fp)
plt.show()
