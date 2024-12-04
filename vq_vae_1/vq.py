import torch
import torch.nn as nn
import torch.nn.functional as F 

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
        
        
      
# # B,c,h,w  
# t = torch.randn([32,256,64,64])
# model = VectorQuantization(4,256,0.1)
# out,loss = model(t)
# print(out.shape,loss)