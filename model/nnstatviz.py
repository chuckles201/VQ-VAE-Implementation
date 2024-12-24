import torch
import torch.nn as nn
from their_vqvae import VQVAE
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import time
import os
'''Viewing the gradients
of our neural net. to see
why our training is so unstable...

1. Log and track grads
of all parameters
2. Track in_grad of each
layer to see how grad is
chaning over time.
Track mean and std.

We will have a dictionary
that saves for each layer (keys)
a grad_mean, and grad_std for each
iteration/batch in our training loop.

-> (type_layer,layers,std&mean&params,iter)'''


# FIX THE FACT THAT WE DON'T NEED HOOKS!
# THE BACKWARD HOOKS ONLY COMPUTE INTERMEDIATE
# GRADIENTS, NOT THE FINAL ONES (WHICH WE CARE ABOUT
# AND ARE EASIER TO INTERPRET!)

'''Function takes the name
of the module, and returns a function
for the backward hook.

Using none when there is no
param/in-gradient for a layer'''
class Expiriment():
    def __init__(self,model):
        self.model = model
        self.gradients = {}
        
        # for each layer-type
        self.gradients["conv_layers"] = {}
        self.gradients["act_layers"] = {}
        self.gradients["norm_layers"] = {}
        self.hook_handles = [] # for removing
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
    
    def save_gradient_hooks(self):
        with torch.no_grad():
            def save_gradients(dict_loc):
                # hook for model
                def hook(module,grad_in,grad_out):
                    # if conv. layer
                    if len(grad_in) == 3:
                        grad_in,weight_grad,bias_grad = grad_in
                    else:
                        grad_in = grad_in[0]
                        weight_grad,bias_grad = None,None
                    
                    # if not the first-layer
                    if grad_in is not None:        
                        mean_in = torch.mean(grad_in.detach()).cpu().item()
                        std_in = torch.std(grad_in.detach()).cpu().item()
                    else:
                        mean_in,std_in = None,None
                    
                    # mean/std for bias and weights
                    if weight_grad is not None:
                        param_w_grad_mean = torch.mean(weight_grad).cpu().item()
                        param_w_grad_std = torch.std(weight_grad).cpu().item()
                        
                        param_b_grad_mean = torch.mean(bias_grad).cpu().item()
                        param_b_grad_std = torch.std(bias_grad).cpu().item()
                        
                        # current-value-state
                        param_w_val_mean = torch.mean(module.weight).cpu().item()
                        param_w_val_std = torch.std(module.weight).cpu().item()
                        param_b_val_mean = torch.mean(module.bias).cpu().item()
                        param_b_val_std = torch.std(module.bias).cpu().item()
                        
                        # dividing value by gradient.
                        data_grad_ratio = torch.abs(torch.mean(module.weight / weight_grad)).cpu().item()
                        
                    else: # if no params.
                        param_w_grad_mean = None
                        param_w_grad_std = None
                        
                        param_b_grad_mean = None
                        param_b_grad_std = None 
                        param_w_val_mean = None
                        param_w_val_std = None
                        param_b_val_mean = None
                        param_b_val_std = None
                        data_grad_ratio = None
                    
                    # appending at given location (layer_type,layer,epochs,info)  
                    # appending in order (for each layer):
                    # 1. mean in
                    # 2. std in
                    # 3. param w grad mean
                    # 4. param w grad std
                    # 5. param b grad mean
                    # 6. param b grad std
                    # and current-state params.
                    self.gradients[dict_loc[0]][dict_loc[1]][0].append(mean_in)
                    self.gradients[dict_loc[0]][dict_loc[1]][1].append(std_in)
                    self.gradients[dict_loc[0]][dict_loc[1]][2].append(param_w_grad_mean)
                    self.gradients[dict_loc[0]][dict_loc[1]][3].append(param_w_grad_std)
                    self.gradients[dict_loc[0]][dict_loc[1]][4].append(param_b_grad_mean)
                    self.gradients[dict_loc[0]][dict_loc[1]][5].append(param_b_grad_std)
                    
                    self.gradients[dict_loc[0]][dict_loc[1]][6].append(param_w_val_mean)
                    self.gradients[dict_loc[0]][dict_loc[1]][7].append(param_w_val_std)
                    self.gradients[dict_loc[0]][dict_loc[1]][8].append(param_b_val_mean)
                    self.gradients[dict_loc[0]][dict_loc[1]][9].append(param_b_val_std)
                    
                    self.gradients[dict_loc[0]][dict_loc[1]][10].append(data_grad_ratio)
                    
                    
                    
                return hook
                
        ### registering with model ###
        self.model = self.model.to('cuda')
        for name, module in self.model.named_modules():
            # checking
            if isinstance(module,(nn.LayerNorm,nn.GroupNorm)):
                print(f'name: {name}, module:{module} **NORM')
                print("Registering hook.")
                self.gradients['norm_layers'][name] = []
                module.register_backward_hook(save_gradients(['norm_layers',name]))
                
                # registering categories of
                # info we'll be saving
                for i in range(11):
                    self.gradients['norm_layers'][name].append([])
                
            elif isinstance(module,(nn.ReLU)):
                print(f'name: {name}, module:{module} **activation')
                print("Registering hook.")
                
                self.gradients['act_layers'][name] = []
                module.register_backward_hook(save_gradients(['act_layers',name]))
                
                for i in range(11):
                    self.gradients['act_layers'][name].append([])
                
            elif isinstance(module,(nn.Conv2d, nn.ConvTranspose2d)):
                print(f'name: {name}, module:{module} **CONV/Tranpose')
                print("Registering hook.")
                
                self.gradients['conv_layers'][name] = []
                module.register_backward_hook(save_gradients(['conv_layers',name]))
                
                
                
                for i in range(11):
                    self.gradients['conv_layers'][name].append([])
                
                    
        print("Gradient hooks saved.")
               
    '''TRAINING model
    here, we run our model
    through some training loops, we'll
    use a standard training loop,
    which can be modified or done manually.
    
    Custom-made for VQ-VAE'''
    def train_model(self,dataset,batch_size=16,iters=1000):
        # get batch function
        def get_batch(data,bs=batch_size):
            indices = torch.randint(0,len(data),size=(bs,))
            data = torch.stack([data[i][0] for i in indices],dim=0)
            
            return data.to('cuda')

        # THEIR PARAMS.
        self.model.to('cuda').train()
        loss_func = nn.MSELoss()
        # less for adaptive-training!
        optimizer = torch.optim.AdamW(params=self.model.parameters(),lr=2e-4,betas=(0.9,0.999),weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100,0.99)
        

        ####### Training LOOP######################
        print("training...")
        start = time.time()
        losses = []
        for i in range(iters):
            batch = get_batch(dataset)
            
            # B,c,h,w
            x,loss_1 = self.model(batch)
            
            # backward-pass
            loss = loss_func(x,batch) + loss_1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),3)
            optimizer.step()
            scheduler.step() # for decay
            
            torch.cuda.synchronize()
            losses.append(loss.item())
            
            if i%10 ==0:
                end = time.time()-start
                print(f"Iter: {i} | Time: {end} | (Avg) Loss:{np.mean(losses[i-10:i])}")
                start = time.time()
            
        
       # clearning hooks
        for name,module in self.model.named_modules():
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()
                
        torch.save(self.model.state_dict(),'weights.pt')

            
            
    
    
    '''Plotting saved-grads

    Shape: (type_layer,layers,epoch,info)
    for each type of layer have a graph of all-layers
    and the corresponding inforamtion.'''
    def plot_gradients_over_time(self,size_in_inches=[30,30]):
        # creating fig. for each
        # type of layer with 3
        # options: in, param_w,param_b
        gradient_dict = self.gradients
        fig,axes = plt.subplots(len(gradient_dict.keys()),6)
        
        layer_types = list(gradient_dict.keys())
        
        fig.set_size_inches(size_in_inches)
        
        for i,layer in enumerate(layer_types):
            # now going into module
            # go over each iter and 
            # add this to graph...
            modules = list(gradient_dict[layer].keys())
            
            # saving all of info...
            # for each module: (modules,epochs,6)
            indiv_layers = [gradient_dict[layer][modules[ind]] for ind in range(len(modules))]
            # for each layer, plotting
            # 6 params stored
            for a,l in enumerate(indiv_layers):
                # this is list 6 parts of indiv layer
                saved_vals = np.array(l)
                saved_vals = np.where(saved_vals == None,np.nan,saved_vals)
                
                # in mean/std
                axes[i][0].errorbar(torch.arange(0,len(saved_vals[0])),
                                    saved_vals[0],yerr=saved_vals[1],
                                    label=f"{modules[a]}")
                axes[i][0].set_title(f"{layer} in_gradients")
                axes[i][0].legend()
                
                # weights grad mean/std
                axes[i][1].errorbar(torch.arange(0,len(saved_vals[2])),
                                    saved_vals[2],yerr=saved_vals[3],
                                    label=f"{modules[a]}")
                axes[i][1].set_title(f"{layer} weight_gradients")
                axes[i][1].legend()
                
                # bias grad mean/std
                axes[i][2].errorbar(torch.arange(0,len(saved_vals[4])),
                                    saved_vals[4],yerr=saved_vals[5],
                                    label=f"{modules[a]}")
                axes[i][2].set_title(f"{layer} bias_gradients")
                axes[i][2].legend()
                
                
                # weight / bias values
                axes[i][3].errorbar(torch.arange(0,len(saved_vals[6])),
                                    saved_vals[2],yerr=saved_vals[7],
                                    label=f"{modules[a]}")
                axes[i][3].set_title(f"{layer} weight_values")
                axes[i][3].legend()
                
                # bias grad mean/std
                axes[i][4].errorbar(torch.arange(0,len(saved_vals[8])),
                                    saved_vals[4],yerr=saved_vals[9],
                                    label=f"{modules[a]}")
                axes[i][4].set_title(f"{layer} bias_values")
                axes[i][4].legend()
                
                # grad-data ratio (only weight)
                # average value proprotion...
                axes[i][5].errorbar(torch.arange(0,len(saved_vals[10])),
                                    saved_vals[10],yerr=0,
                                    label=f"{modules[a]}")
                axes[i][5].set_title(f"{layer} grad-to-data-ration")
                axes[i][5].legend()
                
            
        plt.show()
        
    def plot_current(self):
        pass
    