import torch
import torch.nn as nn
from vq import VectorQuantization



'''Implementing their architecture
as described in the VQ-VAE paper.

Trying to understand their design-decisions,
and how the network is created.

Their architecture:
1. vary dimensionality of embeddings and
the number of available embeddings

2. Encoder:
- 2 strided convolutional layers,
stride=2, winddow 4x4, 
- 2 residual 3x3 blocks (relu 3x3 conv.
relu 1x1 conv)

- all with 256 hidden-units (for both?)

3. Decoder: 2 residual blocks,
2 stransposed convolutions with stride 2,
and 4x4 window (same as encoder)


------
The idea is that first we downsample, and then
we learn from the images with our residual-blocks.
'''



'''Residual block

in paper they intialize weights with 
sqrt(2)/sqrt(nin), and use RELU activations,
as these were found to train faster for CNNs.

When the dimensions mismatch, we use a stride-2
1x1 convolution to downsample the residual connections.

However, in this paper the dimensions don't mismatch,
and there is a 1x1 convolution used to help
the model 'mix' features from different maps (add)
them up????

This is essentially a feed-forward layer, and adding
them across multiple-channels.

'''
class ResidualBlock(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # conv_1 3x3, relu
        # conv_2 1x1, allows us
        # to mix feature-maps
        self.layers = nn.Sequential(
            #nn.GroupNorm(32,h_dim),
            nn.ReLU(),
            nn.Conv2d(h_dim,h_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            #nn.GroupNorm(32,h_dim),
            nn.Conv2d(h_dim,h_dim,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )
        
        
        ### intiializating weights ###
        # default init'd 1/sqrt(n) unif.
        
        # nn.init.kaiming_normal_(self.layers[1].weight,nonlinearity='relu')
        # nn.init.kaiming_normal_(self.layers[3].weight,nonlinearity='relu')
        
        # nn.init.zeros_(self.layers[1].bias)
        # nn.init.zeros_(self.layers[3].bias)
        
    def forward(self,x):
        resid = x
        x = self.layers(x)
        return x + resid
    
    
    
'''Downsampling Block

This is a block for decreasing the
spatial dimensions, and is a simple 
convolution with a stride of 2

padding of 1 to maintain dims.'''
class DownBlock(nn.Module):
    def __init__(self, h_dim,in_dim):
        super().__init__()
        self.layers = nn.Conv2d(in_dim,h_dim,
                      kernel_size=4,
                      stride=2,
                      padding=1)

        ### intiializating weights ###
        # default: unif +/- 1/sqrt(n)
        nn.init.kaiming_normal_(self.layers.weight,nonlinearity='linear')
        nn.init.zeros_(self.layers.bias)
        
    def forward(self,x):
        x = self.layers(x)
        return x
    
   
'''Encoder, as per the paper
is 2 downblocks and 2 res-blocks
(1/4 original dimensionality.)''' 
class Encoder(nn.Module):
    def __init__(self,in_dim,h_dim):
        super().__init__()
        self.layers = nn.Sequential(
            DownBlock(in_dim=in_dim, h_dim=h_dim),
            #nn.GroupNorm(32,h_dim),
            DownBlock(in_dim=h_dim,h_dim=h_dim),
            ResidualBlock(h_dim),
            ResidualBlock(h_dim)
        )
        
    def forward(self,x):
        return self.layers(x)
    
    
    
### Decoder ####    
'''Decoder Block
will be the exact opposite
of our encoder (symmetric), and we
will include transposed-convolutions (reverse)
convolutions.

Here we'll build the up (transposed conv)
block which should double the dimensionality
with a convolution...

Conv. transposed block is literally just a 
way to get the 'reverse' convolution, where all
of the connected layers that would have been
in the 'forward' convolution are connected
in the 'backward one' (same as the backward pass).'''

class UpBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            kernel_size=4,
            padding=1
        )
        
        ## Init layers with linear  init... ##
        
        nn.init.kaiming_normal_(self.transpose.weight,nonlinearity='linear')
        nn.init.zeros_(self.transpose.bias)
        
    def forward(self,x):
        return self.transpose(x)
    

'''The decoder is now made
up of our residual blocks, and then the 
rebuilding with our conv-blocks.

The idea is that we gain 'useful' information
in our residual blocks (non-linearities for 
expressivity), and then rebuild this with linear
transformations in upblock!'''

class Decoder(nn.Module):
    def __init__(self, in_channels,h_dim):
        super().__init__()
        self.decode_layers = nn.Sequential(
            ResidualBlock(h_dim),
            ResidualBlock(h_dim),
            UpBlock(h_dim,h_dim),
            #nn.GroupNorm(32,h_dim),
            UpBlock(h_dim,in_channels)
        )
        
    def forward(self,x):
        return self.decode_layers(x)
    

###################
# full VQ-VAE!
'''Now, creating full VQ-VAE
with incorporated loss-function.

We've already defined the codebook-snapping,
so we just plug our output in and receive the codebook
vector in our bottleneck!'''
class VQVAE(nn.Module):
    def __init__(self,in_dim,h_dims,n_embeddings,beta):
        super().__init__()
        self.encoder = Encoder(in_dim,h_dims)
        self.decoder = Decoder(in_dim,h_dims)
        # embeddings match h-dim.
        # selects for each pixel the corresp.
        # closest based on all channels.
        self.vq = VectorQuantization(n_embeddings,h_dims,beta)
        
    def forward(self,x):
        encoded = self.encoder(x)
        vq_assign,loss_1 = self.vq(encoded)
        out = self.decoder(vq_assign)
        return out,loss_1

###########################
# # testing
# t = torch.randn([32,3,128,128]).to('cuda')
# vqvae = VQVAE(3,256,400,0.25).to('cuda')
# out,temp_loss =vqvae(t)
# print(out.shape,temp_loss)