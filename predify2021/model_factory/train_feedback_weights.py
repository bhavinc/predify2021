#%%
#########################
# In this script we train p-EfficientNets on ImageNet
# We use the pretrained model and only train feedback connections.
#########################
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from .get_model import get_model



################################################
#       Global configs
################################################

class Args():

    def __init__(self):
        self.random_seed = None                    #random_seed for the run
        self.batchsize = 64                        #batchsize for training
        self.num_workers = 8                       #number of workers
        self.num_epochs = 50                       #number of epochs
        self.start_epoch = 1

        self.log_dir='./testtbd_runs_train_feedbacks'       #tensorboard logdir
        self.task_name =  'pefficientnet_b0_lr0.001RMProp_with_cosingannealing'       #dir_name
        self.extra_stuff_you_want_to_add_to_tb = 'same_cosine_annealing_with_t0_3'

        self.imagenet_dir = '/path/to/imagenet/'
        self.optim_name = 'RMSProp'
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.ckpt_every = None   #TODO

        # optional
        self.resume = None                         #resuming the training 
        self.resume_ckpts= None                    #path to the checkpoints. Should be a list of len equal to NUMBER_OF_PCODERS


args = Args()




if args.random_seed:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

if not os.path.exists(args.task_name):
    print (f'Creating a new dir : {args.task_name}')
    os.mkdir(args.task_name)




################################################
#          Net , optimizers
################################################

pnet = get_model('peffb0',deep_graph=False)
pnet.to(device)

NUMBER_OF_PCODERS = pnet.number_of_pcoders

loss_function = nn.MSELoss()
optimizer = optim.RMSprop([{'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters()} for x in range(NUMBER_OF_PCODERS)],
                        lr=args.lr,
                        weight_decay=args.weight_decay
                    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3)

################################################
#       Dataset and train-test helpers
################################################
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


train_ds     = ImageNet(args.imagenet_dir, split='train', download=False, transform=transform_val)
train_loader = torch.utils.data.DataLoader(train_ds,  batch_size=args.batchsize, shuffle=True, drop_last=False,num_workers=args.num_workers,pin_memory=True)

val_ds     = ImageNet(args.imagenet_dir, split='val', download=False, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_ds,  batch_size=args.batchsize, shuffle=True, drop_last=False,num_workers=args.num_workers,pin_memory=True)



def train_pcoders(net, epoch, writer,train_loader,verbose=True):

    ''' A training epoch '''
    
    net.train()

    tstart = time.time()
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()
        images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
            if writer is not None:
                writer.add_scalar(f"MSE Train/PCoder{i+1}", a.item(), epoch * len(train_loader) + batch_index)

        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batchsize + len(images),
            total_samples=len(train_loader.dataset)
        ))
        print ('Time taken:',time.time()-tstart)
        if writer is not None:
            writer.add_scalar(f"MSE Train/Sum", loss.item(), epoch * len(train_loader) + batch_index)


def test_pcoders(net, epoch, writer,test_loader,verbose=True):

    ''' A testing epoch '''

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(net.number_of_pcoders)]
    for batch_index, (images, _) in enumerate(test_loader):
        net.reset()
        images = images.cuda()
        with torch.no_grad():
            outputs = net(images)
        for i in range(net.number_of_pcoders):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()
    
    loss_sum = 0
    for i in range(net.number_of_pcoders):
        final_loss[i] /= len(test_loader)
        loss_sum += final_loss[i]
        if writer is not None:
            writer.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], epoch * len(test_loader))
    if writer is not None:
        writer.add_scalar(f"MSE Test/Sum", loss_sum, epoch * len(test_loader))

    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss_sum,
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        trained_samples=batch_index * args.batchsize + len(images),
        total_samples=len(test_loader.dataset)
    ))
    print ('Time taken:',time.time()-tstart)





################################################
#        Load checkpoints if given...
################################################

if args.resume:

    assert len(args.resume_ckpts) == NUMBER_OF_PCODERS ; 'the number os ckpts provided is not equal to the number of pcoders'

    print ('-'*30)
    print (f'Loading checkpoint from {args.resume_ckpts}')
    print ('-'*30)

    for x in range(NUMBER_OF_PCODERS):
        checkpoint = torch.load(args.resume_ckpts[x])
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        getattr(pnet,f"pcoder{x+1}").pmodule.load_state_dict({k[len('pmodule.'):]:v for k,v in checkpoint['pcoderweights'].items()})

    print ('Checkpoint loaded...')

else :
    print ("Training from scratch...")



# summarywriter
sumwriter = SummaryWriter(f'{args.log_dir}/{args.task_name}', filename_suffix=f'')
optimizer_text = f"Optimizer   :{args.optim_name}  \n lr          :{optimizer.defaults['lr']} \n batchsize   :{args.batchsize} \n weight_decay:{args.weight_decay} \n {args.extra_stuff_you_want_to_add_to_tb}"
sumwriter.add_text('Parameters',optimizer_text,0)


################################################
#              Train loops
################################################
for epoch in range(args.start_epoch, args.num_epochs):
    train_pcoders(pnet, epoch, sumwriter,train_loader)

    test_pcoders(pnet, epoch, sumwriter,val_loader)
    

    for pcod_idx in range(NUMBER_OF_PCODERS):
        torch.save({
                    'pcoderweights':getattr(pnet,f"pcoder{pcod_idx+1}").state_dict(),
                    'optimizer'    :optimizer.state_dict(),
                    'epoch'        :epoch,
                    }, f'{args.task_name}/pnet_pretrained_pc{pcod_idx+1}_{epoch:03d}.pth')

    # print (f'Time taken : {tend-tstart} \n\n\n\n')


