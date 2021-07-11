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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


training_data = datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), 
        (1.0,1.0,1.0))]))
validation_data = datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), 
        (1.0,1.0,1.0))]))
  
data_variance = np.var(training_data.data / 255.0)

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

class Residual(nn.Module):
 
    def __init__(self,in_channels,num_hiddens,num_residual_hiddens):
        super().__init__()

        self._block = nn.Sequential(nn.ReLU(True),
        nn.Conv2d(in_channels=in_channels,
        out_channels=num_residual_hiddens,
        kernel_size=3, stride=1, padding=1, bias=False),nn.ReLU(True),
        nn.Conv2d(in_channels=num_residual_hiddens,
        out_channels=num_hiddens, kernel_size=1, stride=1, bias=False))
  
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,
        num_residual_hiddens):
        super().__init__()

        self._num_residual_layers = num_residual_layers

        self._layers = nn.ModuleList([
        Residual(in_channels, num_hiddens, num_residual_hiddens)
        for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
          x = self._layers[i](x)
        return F.relu(x)
        
class Encoder(nn.Module):
  def __init__(self,in_channels,num_hiddens,
  num_residual_layers,num_residual_hiddens):
    super().__init__()
    
    self._conv_1 = nn.Conv2d(in_channels=in_channels,
                             out_channels=num_hiddens//2,
                             kernel_size=4, stride=2, padding=1)
    self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                             out_channels=num_hiddens,
                             kernel_size=4, stride=2, padding=1)
    self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                             out_channels=num_hiddens,
                             kernel_size=3, stride=1, padding=1)
    self._residual_stack = ResidualStack(in_channels=num_hiddens,
    num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,
    num_residual_hiddens=num_residual_hiddens)
    
  def forward(self, inputs):
    x = self._conv_1(inputs)
    x = F.relu(x)
    
    x = self._conv_2(x)
    x = F.relu(x)
    
    x = self._conv_3(x)
    return self._residual_stack(x)

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, num_hiddens, 
        num_residual_layers, num_residual_hiddens):
        super().__init__()
    
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
        out_channels=num_hiddens,kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
        num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
        out_channels=num_hiddens//2,kernel_size=4, stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(
        in_channels=num_hiddens//2,out_channels=out_channels,
        kernel_size=4, stride=2, padding=1)
        
    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)
        
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
 
    def forward(self, inputs):
        # convert inputs from (B, C, H, W) to (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
     
        # flatten input to 2D (B * H * W , C)
        flat_input = inputs.view(-1, self._embedding_dim)
     
        # calculate distances (euclidean)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
          + torch.sum(self._embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
     
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],
        self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # 針對axis=1的地方，將encoding_indices中的每個index位置改為1
     
        # Quantize and unflaten
        quantized = torch.matmul(encodings, 
          self._embedding.weight).view(input_shape)
     
        # Loss .detach()這個method是關鍵
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # detach()
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # detach()
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
     
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
          torch.log(avg_probs + 1e-10)))
     
        # convert quantized from (B, H, W, C) to (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2)
      
        return loss, quantized.contiguous(), perplexity, encodings

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, 
        num_residual_hiddens, num_embeddings, embedding_dim, 
        commitment_cost, decay=0):

        super().__init__()

        self._encoder = Encoder(3, num_hiddens, num_residual_layers, 
        num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
        out_channels=embedding_dim, kernel_size=1, stride=1)

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, 
        commitment_cost)

        self._decoder = Decoder(embedding_dim, 3, num_hiddens, 
        num_residual_layers, num_residual_hiddens)
 
    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


if __name__ == "__main__":

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    training_loader = DataLoader(training_data, batch_size = batch_size, shuffle = True, pin_memory = True)
    validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = True, pin_memory = True)
    train_res_recon_error = []
    train_res_perplexity = []
    
    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
  
        if (i+1) % 100 ==0:
            print('{:d} iterations, recon_error : {:.3f}, perplexity: {:.3f}\r\n'.format(i+1, np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))

    PATH='saved_models/vqvae_params.pkl'
    torch.save(model.state_dict(), PATH)
