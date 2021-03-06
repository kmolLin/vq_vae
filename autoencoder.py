import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2

from vqvae import CustomDataset
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchsummary import summary


def grab_cam_vis(image, model):
    """

    :param image: tensor of image shape (3, 512, 512)
    :param model: autoencoder model
    :return:
    """
    model.to(device)
    model.eval()

    image = image.to(device)
    image = image.unsqueeze(0)

    cam = GradCAM(model=model, target_layer=model.decoder[-1], use_cuda=True)

    def segmentation_get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category, :, :].mean()
        return loss

    cam.get_loss = segmentation_get_loss
    target_category = 281

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image, target_category=1, aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]

    a = val[num_image].permute(1, 2, 0).numpy()

    visualization = show_cam_on_image(a, grayscale_cam)

    plt.imshow(visualization)
    plt.show()



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Sequential(self.conv1, self.relu, self.pool, self.conv2, self.relu, self.pool)
        self.decoder = nn.Sequential(self.t_conv1, self.relu, self.t_conv2, self.sigmoid)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        x = self.encoder(x)
        # x = self.pool(x)
        x = self.decoder(x)
        #
        # x = F.relu(self.t_conv1(x))
        # x = F.sigmoid(self.t_conv2(x))

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    batch_size = 16

    training_data = CustomDataset("image_d/train", transform=transforms.Compose(
        [transforms.ToTensor()]))

    val = CustomDataset("image_d/test", transform=transforms.Compose(
        [transforms.ToTensor()]))

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    model = ConvAutoencoder()
    model.to(device)
    summary(model, (3, 1024, 1280))
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    n_epochs = 500
    # model.train()
    # for epoch in range(1, n_epochs + 1):
    #     # monitor training loss
    #     train_loss = 0.0
    #
    #     # Training
    #     for data in training_loader:
    #         images = data
    #         images = images.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, images)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item() * images.size(0)
    #
    #     train_loss = train_loss / len(training_loader)
    #     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    #
    # torch.save(model.state_dict(), "saved_models/autoencoder.pkl")

    model.load_state_dict(torch.load("saved_models/autoencoder.pkl"))


    num_image = 3
    image = val[num_image]
    grab_cam_vis(image, model)

    image = image.to(device)
    image = image.unsqueeze(0)

    outputs = model(image)
    image = outputs.cpu().clone()
    image = image.squeeze(0)
    new_img_PIL = transforms.ToPILImage()(image)
    new_img_PIL.show()