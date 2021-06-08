# predify2021

Code for reproducing the results presented in the paper 'Predify:Augmenting deep neural networks with brain-inspired predictive coding dynamics' https://arxiv.org/pdf/2106.02749.pdf


```
Dependencies :
- predify :  https://github.com/miladmozafari/predify
- torch 
- tensorboard
- loguru

- timm : https://github.com/rwightman/pytorch-image-models (for EfficientNetB0)
- foolbox 3.x (for Adversarial Attacks)
```


## Repository structure
-  `model_factory` provides predified versions of the models (VGG and EfficientNetB0) using `get_model` function. 
-  `adversarial_attacks` contains all scripts for performing and analysing adversarial attacks performed in the paper.
-  `mCE_scores` contains scripts for performing and calculating mCE scores on the predified networks.
-  `manifold_projection` contains scripts for calculating the correlation distances between clean and noisy representations (refer paper for more details).

