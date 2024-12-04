import torch
import torch.nn as nn
from torch.nn import functional as F 
import time

import vae,vq
from vae import VQVAE

import importlib
import os

importlib.reload(vae)


# hyper-params
torch.cuda.empty_cache()
n_embed = 512
iters = 100
beta = 0.256
batch_size=16
device='cuda'

model = VQVAE(device=device)
path = os.path.join('weights.pt')
model.load_state_dict(torch.load(path))
model.to(device)

print("Parameters:")
print(sum([p.numel() for p in model.parameters()]))



def get_batch(data,bs):
    indices = torch.randint(0,len(data),size=(bs,))
    data = torch.stack([data[i][0] for i in indices],dim=0)
    
    return data.to(device)


# downloading data
import torchvision
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256,256]),
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.Imagenette(
    './data/imagenette',
    split = 'train',
    download=False,
    transform=transform
)

model.train()
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(),lr=2e-4,betas=(0.9,0.999),weight_decay=1e-4)


####### Training LOOP######################
print("training...")
losses = []
for i in range(iters):
    start = time.time()
    batch = get_batch(train_data,batch_size)
    
    # B,c,h,w
    x,loss_1 = model(batch)
    
    # backward-pass
    loss = loss_func(x,batch) + loss_1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    end = time.time()-start
    
    losses.append(loss.item())
    print(f"Iter: {i} | Time: {end} | Loss:{loss.item()}")
    
    


# saving models final weights
torch.save(model.state_dict(),"weights.pt")

# then call load_state_dict when 
# your'e ready to eval!