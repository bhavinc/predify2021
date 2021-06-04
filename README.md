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


`Repository structure`
`model_factory` package gives predified versions of the models using `get_model` function. It is used in ther experiments.
`adversarial_attacks` contains all scripts for performing and analysing adversarial attacks.
`mCE_scores` contains all scripts for performing and calculating mCE scores on the predified networks.
`manifold_projection` contains all scripts for calculating the correlation distances between clean and noisy representations.

