import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from torch.utils.data import Dataset
import os

from six.moves import xrange

import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchsummary import summary
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        im = im.resize((256, 256))
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
            im = im.resize((256, 256))
            im = np.array(im)
            tmp.append(im)
        data_variance = np.var(np.array(tmp) / 255.0)
        return data_variance


## Load Data

# training_data = datasets.CIFAR10(root="data", train=True, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#                                   ]))
#
# validation_data = datasets.CIFAR10(root="data", train=False, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#                                   ]))
training_data = CustomDataset("imagess", transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))

validation_data = CustomDataset("val_data", transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))

# data_variance = np.var(training_data.data / 255.0)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

## Encoder & Decoder Architecture

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
        
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
        
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


if __name__ == "__main__":
    batch_size = 256
    num_training_updates = 15000

    num_hiddens = 128
    num_residual_hiddens = 64
    num_residual_layers = 4

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-3

    data_variance = training_data._get_all_data()

    training_loader = DataLoader(training_data,
                             batch_size=4,
                             shuffle=True,
                             pin_memory=True)
    validation_loader = DataLoader(validation_data,
                                   batch_size=1,
                                   shuffle=True,
                                   pin_memory=True)
                                   
    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # model.train()
    # train_res_recon_error = []
    # train_res_perplexity = []
    #
    # for i in range(num_training_updates):
    #     data = next(iter(training_loader))
    #     data = data.to(device)
    #     optimizer.zero_grad()
    #
    #     vq_loss, data_recon, perplexity = model(data)
    #     recon_error = F.mse_loss(data_recon, data) / data_variance
    #     loss = recon_error + vq_loss
    #     loss.backward()
    #
    #     optimizer.step()
    #
    #     train_res_recon_error.append(recon_error.item())
    #     train_res_perplexity.append(perplexity.item())
    #
    #     if (i+1) % 100 == 0:
    #         print('%d iterations' % (i+1))
    #         print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
    #         print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
    #         print()
    #         if np.mean(train_res_recon_error[-100:]) < 0.009:
    #             value = np.mean(train_res_recon_error[-100:])
    #             torch.save(model.state_dict(), f"saved_models/high_cnn/vqvae_{i+1}_{value:.4f}.pkl")
    # torch.save(model.state_dict(), f"saved_models/t1t1_vqvae.pkl")
    # train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    # train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    #
    # f = plt.figure(figsize=(16,8))
    # ax = f.add_subplot(1,2,1)
    # ax.plot(train_res_recon_error_smooth)
    # ax.set_yscale('log')
    # ax.set_title('Smoothed NMSE.')
    # ax.set_xlabel('iteration')
    #
    # ax = f.add_subplot(1,2,2)
    # ax.plot(train_res_perplexity_smooth)
    # ax.set_title('Smoothed Average codebook usage (perplexity).')
    # ax.set_xlabel('iteration')
    # plt.show()
    #
    # exit()

    PATH = 'saved_models/t1t1_vqvae.pkl'
    model.load_state_dict(torch.load(PATH))

    model.eval()

    valid_originals = next(iter(training_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, pr, enc = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)
    avg_probs = torch.mean(enc, dim=0)

    train_originals = next(iter(training_loader))
    train_originals = train_originals.to(device)
    _, train_reconstructions, _, _ = model._vq_vae(train_originals)


    def show(img):
        npimg = img.numpy()
        fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
    show(make_grid(valid_reconstructions.cpu().data)+0.5, )
    plt.show()
    show(make_grid(valid_originals.cpu()+0.5))
    plt.show()

    # proj = umap.UMAP(n_neighbors=3,
    #          min_dist=0.1,
    #          metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
    # plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
    # plt.show()