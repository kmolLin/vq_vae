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
from torch.utils.data import Dataset
import os
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
 
    def __init__(self,in_channels, num_hiddens, num_residual_hiddens):
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
    num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
    num_residual_hiddens=num_residual_hiddens)
    
  def forward(self, inputs):
    x = self._conv_1(inputs)
    x = F.leaky_relu(x)
    
    x = self._conv_2(x)
    x = F.leaky_relu(x)
    
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
        # print(flat_input.size())
     
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


class CustomDataset(Dataset):
    # im_name_list, resize_dim,
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.im_list = os.listdir(self.root_dir)
        # self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        # im = im.crop((600, 0, 1700, 1400))
        # im = im.resize((512, 512))
        # im.show()
        # im = np.array(im)
        # im = Image(im, self.resize_dim, interp='nearest')
        # im = im / 255.0

        if self.transform:
            im = self.transform(im)

        return im

    def _get_all_data(self):
        tmp = []
        for im in self.im_list:
            im = Image.open(os.path.join(self.root_dir, im))
            # im = im.resize((512, 512))
            im = np.array(im)
            tmp.append(im)
        data_variance = np.var(np.array(tmp) / 255.0)
        return data_variance


if __name__ == "__main__":

    brightness_change = transforms.ColorJitter()
    rotate = transforms.RandomRotation(30)

    transform_set = [rotate, brightness_change]
    mean = [0.5, 0.5, 0.5]
    std = [1.0, 1.0, 1.0]

    training_data = CustomDataset("imagess", transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std),
             # transforms.RandomApply(transform_set, p=0.5)
             ]))

    # validation_data = CustomDataset("image_data/train", transform=transforms.Compose(
    #         [transforms.ToTensor()]))

    data_variance = training_data._get_all_data()
    # training_data = datasets.CIFAR10(
    #         root='data', train=True, download=True,
    #         transform=transforms.Compose(
    #         [transforms.ToTensor()]))
    # validation_data = datasets.CIFAR10(
    #         root='data', train=False, download=True,
    #         transform=transforms.Compose(
    #         [transforms.ToTensor()]))

    # print(training_data.data.shape)
    # data_variance = np.var(training_data.data / 255.0)

    torch.cuda.manual_seed(123456)
    batch_size = 12
    num_training_updates = 15000
    epoch = 15000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 0.02

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle = True, pin_memory = True)
    # validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = True, pin_memory = True)
    train_res_recon_error = []
    train_res_perplexity = []
    val_score = 1000

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for i in range(100):
        for batch_idx, data in enumerate(training_loader):
            # data = next(iter(training_loader))
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if batch_idx % 100 == 0:
                print('{:d} epoch, recon_error : {:.3f}, perplexity: {:.3f}\r\n'.format(i+1, np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))
                if val_score > np.mean(train_res_recon_error[-100:]):
                    val_score = np.mean(train_res_recon_error[-100:])
                    if i == 0:
                        continue
                    elif val_score < 0.01:
                        torch.save(model.state_dict(), f"saved_models/epoch{i}_{val_score:.2f}.pkl")
                    # elif i > 200:
                    #     torch.save(model.state_dict(), f"saved_models/epoch{i}_{val_score:.2f}.pkl")
                    #     exit()
        # scheduler.step()

    PATH = 'saved_models/vqvae_params.pkl'
    torch.save(model.state_dict(), PATH)
