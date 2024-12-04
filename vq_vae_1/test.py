import torch
from vae import VQVAE
import os

path = os.path.join("weights.pt")
model = VQVAE()

torch.save(model.state_dict(),path)