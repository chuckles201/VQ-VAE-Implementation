import torch
import PIL
from dataloader import ImageDataset
import os
from tqdm import tqdm

''' Storing Latents
In this file, we will simply make a function,
that given a VAE, can run every single instance
of the dataset, and save the representations in 
a file.
'''

def store_latents(dataset,path,model):
    # takes entire dataset and maps to 
    # a specific path for latent images.
    # returns the path of the representations
    indxs = []
    latent_outputs = []
    full_path = os.path.join(path,'latent_storage.pt')
    
    # creating folder
    if not os.path.exists(path):
        os.makedirs(path)
    
    # if already there
    if os.path.exists(full_path):
        print("Mapping already exists, skipping latent-storage")
       
    # creating file with latents 
    else:
        # take all images.
        ra = tqdm(range(len(dataset)))
        for i in ra:
            # appending latent-outputs
            # passing-thru labels.
            latent_outputs.append(model(dataset[i][0]))
            
        save = [latent_outputs]
        torch.save(save,full_path)

