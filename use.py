from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from vqvae import Model

batch_size = 256
num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
    interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    
    PATH='saved_models/vqvae_params.pkl'
    model.load_state_dict(torch.load(PATH))
    validation_data = datasets.CIFAR10(
            root='data', train=False, download=True,
            transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), 
            (1.0,1.0,1.0))]))
    validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = True, pin_memory = True)
    
    (valid_originals, _) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)
    show(make_grid(valid_originals.cpu()[:16,:,:,:]+0.5))
    plt.show()
    
# PATH='saved_models/vqvae_params.pkl'
# torch.save(model.state_dict(), PATH)
