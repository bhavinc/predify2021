# predify2021
Code for reproducing the results presented in the paper 'Predify:Augmenting deep neural networks with brain-inspired predictive coding dynamics'

```
Dependencies :

- predify
- torch
- tensorboard
- foolbox 3.x
- loguru
```

The model_factory package gives predified versions of the models using `get_model` function. It is used in ther experiments for calculating the manifold projection (correlation distances), mCE scores, and the adersarial robustness across timesteps. 
