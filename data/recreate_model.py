import torch.nn.functional as F 
import torch
import torch.nn as nn



'''Vector quantization
We'll create our embedding table,
which is simple a D-dimensional table
that is selected from based on the nearest L-2
neighbor.
This should take an input of 
(B,C,H,W) and output
->(B,C,H,W) embeddings 
of same dimensionality.
We want to freeze the gradients during
part of the loss (where sg in formula).
'''

class VectorQuantization(nn.Module):
    # beta controls the commitment
    # cost to the encoder
    def __init__(self,n_embed,dim_embed,beta,device='cuda'):
        
        super().__init__()
        self.device=device
        self.commitment_cost = beta
        self.embeddings = nn.Embedding(n_embed,dim_embed)
        
    def forward(self,x):
        # find similarity based on l2
        # norm for every combo
        # we expand out formula:
        # (z^2+e^2-2e*z), and take the
        # sum of each element for this
        
        # (B,C,H,W) -> (B*H*W,C)
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).contiguous().view(b*h*w,c)
        
        # getting sum -> B*H*W,1
        hw_sum_squared = torch.sum(x**2,dim=-1,keepdim=True)
        # Emb,1
        emb_sum_squared = torch.sum(self.embeddings.weight**2,dim=-1,keepdim=True)
        # bhw, dim @ dim, emb -> bhw,emb dot prodds.
        hwemb_prod = -2*torch.matmul(x,self.embeddings.weight.transpose(0,1))
        
        # bhw, emb
        l2_squared = hwemb_prod + emb_sum_squared.transpose(0,1) + hw_sum_squared
        # choosing max indices: bhw,1
        indices = torch.argmin(l2_squared,dim=-1,keepdim=True)
        
        # creating selected one-hot vector
        one_hot = torch.zeros(b*h*w,self.embeddings.weight.shape[0],device=self.device)
        one_hot.scatter_(dim=1,index=indices,value=1)
        
        # now, selecting part of embedding 
        quantized = one_hot @ self.embeddings.weight
    
        
        # loss objective (squared l2)
        # detatch appropriate
        # gradient from this.
        commitment = self.commitment_cost*F.mse_loss(quantized.detach(),x)
        codebook_close = F.mse_loss(quantized,x.detach())
        
        loss = commitment + codebook_close
        
        # this means all of quantized 
        # will now be the same value, but
        # be connected to x in comp-graph.
        quantized = x + (quantized-x).detach()
        
        # (bhw,c) -> (b,c,h,w)
        quantized = quantized.view(b,h,w,c)
        quantized = quantized.permute(0,3,1,2).contiguous()
        
        # returning our quantized vectors and loss!
        return quantized,loss
        
class ResidualBlock(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # conv_1 3x3, relu
        # conv_2 1x1, allows us
        # to mix feature-maps
        self.layers = nn.Sequential(
            nn.GroupNorm(32,h_dim),
            nn.ReLU(),
            nn.Conv2d(h_dim,h_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.GroupNorm(32,h_dim),
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
            nn.GroupNorm(32,h_dim),
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
            nn.GroupNorm(32,h_dim),
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
    def __init__(self,in_dim=3,h_dims=256,n_embeddings=512,beta=0.25):
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