# predify2021

Code for reproducing the results presented in the paper 'Predify:Augmenting deep neural networks with brain-inspired predictive coding dynamics' (https://arxiv.org/pdf/2106.02749.pdf)


## Dependencies
<pre>
- predify  (<a href="https://github.com/miladmozafari/predify">Repository</a>)
- torch 
- tensorboard
- loguru

For EfficientNetB0:
- timm     (<a href="https://github.com/rwightman/pytorch-image-models">Repository</a>)

For Adversarial Attacks:
- foolbox 3.x
</pre>


## Repository structure
-  `model_factory` provides predified versions of the models (VGG and EfficientNetB0) using `get_model` function. 
-  `adversarial_attacks` contains all scripts for performing and analysing adversarial attacks performed in the paper.
-  `mCE_scores` contains scripts for performing and calculating mCE scores on the predified networks.
-  `manifold_projection` contains scripts for calculating the correlation distances between clean and noisy representations (refer paper for more details).

## Weights of all the models

[Link to the PEfficientNetB0 weights](https://www.dropbox.com/s/0np1pzp3o3qhonv/weights_pefficientNetB0_imagenet.zip?dl=0) 

[Link to the PVGG_Weights](https://www.dropbox.com/s/8lzp6wfo6n3bymk/weights_pvgg16_imagenet.zip?dl=0)
