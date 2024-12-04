import torch
import torch.nn as nn
from torch.nn import functional as F 
from vq import VectorQuantization


'''Residual Block that does
a convolution, with a residual layer.

Specify number of channels, and does
not grow the spatial dimensions.


If the channels are different, the residual connections
are all taken, and each one is added to the others.

The non-linearity will be one that has happened to
work well in practice:
'''


# custom residual block:
# convolution, and residual connection!
class ResBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.GroupNorm(32,out_channels)
        
        # keeps dimensionality!
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.LeakyReLU()
        
        if self.in_channels == self.out_channels:
            self.resid_map = nn.Identity()
        else:
            # mapping with linear-projection
            self.resid_map = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        
    # conv and residual
    def forward(self,x):
        # B,In,H,W -> B,out,h,w (for both)
        residue = self.resid_map(x)
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x + residue



'''Same block, but takes down dimensionality X2

No residual connection used in this layer!!!

Should we use normalization???'''
class DownBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # halves spatial-dims
        self.norm = nn.GroupNorm(32,out_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=4,padding=1,stride=2)
        self.relu = nn.LeakyReLU()
        
    # conv and residual
    def forward(self,x):
        # B,In,H,W -> B,out,h,w (for both)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.norm(x)
        
        return x 
    

'''Encoder for VAE, made up of residual
blocks, with MLPs and downsampling to increase
channel-dims, and decrease spatial-dims.

For our output, we should output
a 2d map of D-dimensional vectors that
match the embedding (to choose) of the
embedding-table.


We will have 
1. Resnet, downsample
2. Repeat until dimensionality acheived


Is there a reason for having a certain config?
''' 

# TODO: groupnorm add?
class Encoder(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        # conv mapping
        self.conv_final = nn.Conv2d(256,256,1,1,0)
        self.relu = nn.LeakyReLU()
        self.layers = nn.Sequential(
            ResBlock(in_channels,128),
            DownBlock(128,128),
            ResBlock(128,128),
            DownBlock(128,256),
            ResBlock(256,256),
            DownBlock(256,256)
            # FINAL: (B,256,H/8,W/8)
        )
        
    def forward(self,x):
        x = self.layers(x)
        x = self.conv_final(x)
        return self.relu(x)
    

##########################################

'''Decoder: will
- select the embedding based on the one-hot
representations
- upsample and decrease channel-dims to match
original image.
'''


class Decoder(nn.Module):
    def __init__(self,out_channels=3):
        super().__init__()
        self.conv_final = nn.Conv2d(64,out_channels,1,1,0)
        self.sig = nn.Sigmoid()
        
        self.layers = nn.Sequential(
            # going backwards, replacing
            # w/ upsamples
            ResBlock(256,256),
            nn.Upsample(scale_factor=2,mode='nearest'),
            ResBlock(256,128),
            ResBlock(128,128),
            nn.Upsample(scale_factor=2,mode='nearest'),
            ResBlock(128,64),
            ResBlock(64,64),
            nn.Upsample(scale_factor=2,mode='nearest'),
            ResBlock(64,64),
            # FINAL: (B,3,H,W)
        )
    
    def forward(self,x):
        x = self.layers(x)
        x = self.conv_final(x)
        x = self.sig(x)
        return x
    


# # testing encoder and decoder

# import time
# # very fast this way!
# device='cuda'        
# t = torch.randn([32,3,256,256],device=device) 
# encoder = Encoder().to(device)
# decoder = Decoder().to(device)

# start = time.time()
# t_out = decoder(encoder(t))
# end = time.time() - start
# print(t_out.shape, "   |    time :",end)

class VQVAE(nn.Module):
    def __init__(self,n_embed=512,d_embed=256,device='cpu'):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vq = VectorQuantization(n_embed,d_embed,beta=0.3,device=device)
    
        
    def forward(self,x):
        x = self.encoder(x)
        x,loss = self.vq(x)
        x = self.decoder(x)
        return x,loss

