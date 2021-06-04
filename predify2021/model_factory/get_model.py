import toml
import torch
import predify




def set_hyperparams(net,hps):

    num = net.number_of_pcoders

    assert len(hps) == num

    for n in range(1,num+1):
        setattr(net,f"ffm{n}",torch.tensor(hps[n-1]['ffm'],dtype=torch.float64))
        setattr(net,f"fbm{n}",torch.tensor(hps[n-1]['fbm'],dtype=torch.float64))
        setattr(net,f"erm{n}",torch.tensor(hps[n-1]['erm'],dtype=torch.float64))


def get_model(name,pretrained=False,deep_graph=False,timesteps=4,hyperparams=None):

    if name in ['pvgg','PVGG']:

        import torchvision
        from .pvgg16_shared import PVGG16SeparateHP , DeepPVGG16SeparateHP    
        vgg = torchvision.models.vgg16(pretrained=True)

        if deep_graph == True:
            pvgg16 = DeepPVGG16SeparateHP(backbone=vgg,number_of_pcoders=5,number_of_timesteps=timesteps ,build_graph=True, random_init=False, ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.01)
        else:
            pvgg16 = PVGG16SeparateHP(backbone=vgg , build_graph=False, random_init=False, ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.01)


        if pretrained == True:
            backward_weight_dir = './'
            print (f'Loading weights from {backward_weight_dir}')
            for n in range(pvgg16.number_of_pcoders):
                checkpoint = torch.load(opj(backward_weight_dir, f'pvgg16_imagenet_pretrained_pc{n+1}_pmodule.pth'))
                getattr(net,f"pcoder{n+1}").pmodule.load_state_dict(checkpoint)
        
        if hyperparams is not None:
            set_hyperparams(pvgg16,hyperparams)

        return pvgg16.eval()

    elif name in ['peffb0','PEfficientNetB0']:


        import timm
        from timm.models import efficientnet_b0
        from .peffficientnetb0_shared import PEfficientNetB0_SeparateHP , DeepPEfficientNetB0_SeparateHP

        net = efficientnet_b0(pretrained=True)

        if deep_graph == True:
            pnet = DeepPEfficientNetB0_SeparateHP(backbone=net,number_of_pcoders=8,number_of_timesteps=timesteps ,build_graph=True, random_init=False, ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.01)
        else : 
            pnet = PEfficientNetB0_SeparateHP(backbone=net , build_graph=False, random_init=False, ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.01)


        if pretrained == True:
            resume_ckpts= [f"./" for x in range(8)]  
            for x in range(8):
                checkpoint = torch.load(resume_ckpts[x])
                getattr(pnet,f"pcoder{x+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items()})

        if hyperparams is not None:
            set_hyperparams(pnet,hyperparams)

        return pnet.eval()

    else:
        raise ValueError('The model name is not supported yet.')
