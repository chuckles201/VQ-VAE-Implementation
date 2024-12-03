import torch
import torch.nn as nn


'''Vector quantization
We'll create our embedding table,
which is simple a D-dimensional table
that is selected from based on the nearest L-2
neighbor.

'''

class VectorQuantization(nn.Module):
    # use torch.argmin to find the nearest
    # vector based on l2 norm; output one-hot
    # to select from decoder-outputs in matmul!
    def __init__(self):
        super().__init__()