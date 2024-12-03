# VQ-VAE implementation

This repo. is an implementation of the VQ-VAE, introduced by [this paper](https://arxiv.org/pdf/1711.00937v2). This is for mostly educational purposes, and the code will be a simple implementation that should be to-the-point, and hopefully my notes can help understand the archtiecture/why and how it works.



# VQ-VAE (1st) Paper notes

Skipping the background, the authors of the paper claim that:
*discrete latent representations, rather than continous ones are more trainable and have better outputs*.

The authors base their architecture off the VAE, which given a latent distribution that is said to be sampled from, and some mapping from the latents to the output image (decoder), we try to infer the encoder (ELBO), while maximizing the likelihood of our data (creating a better model).

Usually, we want to model a continous latent-space, however the authors make arguments for using discrete outputs. For example, discrete outputs can represent concepts like 'hand' or 'dog', while continous latent-variables may lead to bluriness and low-sample variance or posterior collapse (which means all the samples look similar/blurry, and posterior collapse is when the model optimizes the loss objective by not changing the latents, and reconstructing the 'average sample'). These problems however, could intuitively be fixed with discretized outputs.

### Embedding

The authors decide to define the model in the following way:
1. An encoder outputs a latent-vector based on the information in the input (Channels,Height, Width)
2. The 'closest' (MSE) embedding vector is assigned to the latent-variables
3. For audio, images, and videos, there is a 1d, 2d, and 3d feature-space (we'll be dealing with images). More on this later


#### Learning

The fundamental diffference in this model, is the 'selection' that we do when picking our embedding. 

The authors propose to simply copy the gradient of the decoders input (embedding) to the output of our encoder.
> My intuition for this, is that we are basically, over the entire dataset getting an 'estimate' for the gradient, when we swap them like this. Our model is learning what patterns produce better outputs for each type of image, and therefore, how to actually output vectors similar to the output-vector. Overall, we are learning the pattern of how to select vectors given image data.

Now, going back to the '2d' embeddings, here is how we actually select the embeddings for our model:
- We have C x H x W vectors from our convolution, and our channel dimensions will match our embedded-vector dimensions. Therefore, each pixel will choose it's latent representation based on it's channel dimension

Now, we can just define our model, and optimize it with the gradient-copying technique!

Furthermore, we will want to not only train our encoder, but also train our dictionary to be well-suited for our encoder. To do this is simple:
- Take the gradient of the embeddings w.r.t. the l2 loss between the encoder and decoder.
- Note: only take the gradient of the vector that is selected.

> So, we are training our model with gradient estimates from our selection of the embedding, but we are also pushing our embeddings to match our outputs in general. This means, when our model is trained, as a whole, there will be good outputs from the encoder, along with a dictionary (embedding table) that is close/well suited for those outputs.

> So, what we can imagine happening, is the model learns in the encoder for a specific type of image to output similar embeddings, and therefore, a codebook vector will assign itself to that. So, after training, the model should have a variety of codebook vectors that have commonly been used to choose from. This is feasable, because we could imagine different feature-maps of different pixels meaning distinct things. (each pixel is a part of the image)

#### Loss-Function

We can break the loss function into 3 parts:
1. Our evidence-reconstruction objective (decoder), which we inherit from the vanilla-VAE
2. As we discussed previously, the second part of the loss is aimed at making the embeddings match the encoder outputs
3. Encoder-Commitment-loss: if our embeddings grow slower than our encoder parameters, then we could get very large embeddings in the encoder. The authors introduce a 'commitment' loss to make sure that the encoder outputs remain close to an embedding.

(No KL-term for the prior (uniform prior assumed, so only evidence part matters))

![training-objective](loss.png)

This is our loss objective where sg means stop gradient (we only calculate gradient of other non-sg term)

So, for each term we will calculate:
1. The average of the gradients for our commitment loss and VQ loss (for each latent-dimension)
2. 


#### Prior
As previously stated, the model is trained with a uniform prior that does not affect the loss term whatsoever. 

However, what they do is after the VAE is done training, they train an auto-regressive model (wavenet for audio, and PixelCNN for video) to create an appropriate image-generation.

> What this means is: we have no prior to sample from (uniform), so we need to define a model that can build up gradually our images. (like how in stable-diffusion we train the model in latent-space)

For example, we would get the latent-representations of a dataset, and do next-pixel prediction (or another objective like diffusion).

### Model Implementation
Now that we understand the loss objective, and our model, let's implement the VAE!


# VQ-VAE (2nd) Paper Notes



### Expiriments

Although we will be expirimenting with some things like
1. Dimensionality for different types of images/ how well model does for different dimensionalities.
2. Different loss-parameters (beta/others)
3. Other architectural changes (like attn/residual/convolutions) 
4

we will first look at some expirimental results and observations observed by the authors.




