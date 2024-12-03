import torch
import torch.nn as nn
from torch.nn import functional as F 



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
        x = self.relu(x)
        
        return x + residue
    
# test: (B,C,H,W) -> (B,C_out,H,W)
t = torch.randn([32,25,256,256])
t_resblock = ResBlock(25,50)
print(t_resblock(t).shape)


'''Same block, but takes down dimensionality X2

No residual connection used in this layer!!!

Should we use normalization???'''
class DownBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # halves spatial-dims
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=4,padding=1,stride=2)
        self.relu = nn.LeakyReLU()
        
    # conv and residual
    def forward(self,x):
        # B,In,H,W -> B,out,h,w (for both)
        x = self.conv_1(x)
        x = self.relu(x)
        
        return x 
    
# test: (B,C,H,W) -> (B,C_out,H/2,W/2)
t = torch.randn([32,25,256,256])
downblock = DownBlock(25,25)
print(downblock(t).shape) 


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


class Encoder(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        # conv mapping
        self.conv_final = nn.Conv2d(256,256,1,1,0)
        self.relu = nn.LeakyReLU()
        self.layers = nn.Sequential(
            ResBlock(in_channels,128),
            DownBlock(128,128),
            ResBlock(128,256),
            DownBlock(256,256)
            # FINAL: (B,256,H/4,W/4)
        )
        
    def forward(self,x):
        x = self.layers(x)
        x = self.conv_final(x)
        return self.relu(x)
    


# very fast this way!
device='cuda'        
t = torch.randn([32,3,256,256],device=device) 
encoder = Encoder().to(device)
t_out = encoder(t)
print(t.shape)


'''Decoder: will
- select the embedding based on the one-hot
representations
- upsample and decrease channel-dims to match
original image.


'''
