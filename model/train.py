import torch
import torch.nn as nn
from torch.nn import functional as F 
import time

import their_vqvae,vq
from their_vqvae import VQVAE

import importlib
import os
import numpy as np

importlib.reload(their_vqvae)


# hyper-params
torch.cuda.empty_cache()
iters = 100000
batch_size=16
device='cuda'
load_model = True ### BE CAREFUL!##################
data = 'cifar_10'

# THEIR PARAMETERS
model = VQVAE(3,256,n_embeddings=512,beta=0.25)
if load_model:
    path = os.path.join('model/weights.pt')
    model.load_state_dict(torch.load(path))

print("Parameters:")
print(sum([p.numel() for p in model.parameters()]))


def get_batch(data,bs):
    indices = torch.randint(0,len(data),size=(bs,))
    data = torch.stack([data[i][0] for i in indices],dim=0)
    
    return data.to(device)


# downloading data
import torchvision
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([128,128]),
    torchvision.transforms.ToTensor()
])

if data == 'cifar_10':
    train_data = torchvision.datasets.CIFAR10(
        './data/cifar_10',
        train=True,
        download=False,
        transform=transform
    )
else:
    train_data = torchvision.datasets.Imagenette(
    './data/imagenette',
    split='train',
    download=False,
    transform=transform
    
    )


# THEIR PARAMS.
model.to(device).train()
loss_func = nn.MSELoss()
# less for adaptive-training!
optimizer = torch.optim.AdamW(params=model.parameters(),lr=2e-4,betas=(0.9,0.999),weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100,0.5)


####### Training LOOP######################
print("training...")
start = time.time()
losses = []
for i in range(iters):
    batch = get_batch(train_data,batch_size)
    
    # B,c,h,w
    x,loss_1 = model(batch)
    
    # backward-pass
    loss = loss_func(x,batch) + loss_1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step() # for decay
    
    torch.cuda.synchronize()
    losses.append(loss.item())
    
    if i%10 ==0:
        end = time.time()-start
        print(f"Iter: {i} | Time: {end} | (Avg) Loss:{np.mean(losses[i-10:i])}")
        start = time.time()
    
    if i % 200 == 0:
        torch.save(model.state_dict(),"weights.pt")
    

# then call load_state_dict when 
# your'e ready to eval!
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.arange(0,len(losses)),losses)
plt.show()