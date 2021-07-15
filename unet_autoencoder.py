import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from vqvae import CustomDataset
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


class Net(nn.Module):
    def __init__(self, useBN=False):
        super(Net, self).__init__()

        self.conv1 = add_conv_stage(3, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)

        return conv0_out


def train(model, training_loader, data):
    epoch = 150 + 1
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.1)
    for epo in range(0, epoch):
        train_loss = 0.0
        for data in training_loader:
            images = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(training_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epo, train_loss))

        if epo % 30 == 0:
            model.eval()
            image = data[0]
            image = image.to(device)
            image = image.unsqueeze(0)
            outputs = model(image)
            image = outputs.cpu().clone()
            image = image.squeeze(0)
            new_img_PIL = transforms.ToPILImage()(image)
            new_img_PIL.show()

    return model


if __name__ == "__main__":
    torch.cuda.manual_seed(123456)
    model = Net()
    model.to("cuda")
    data = CustomDataset("image_d/train",  transform=transforms.Compose(
        [transforms.ToTensor()]))
    training_loader = DataLoader(data, batch_size=16, shuffle=True, pin_memory=True)
    # traind_model = train(model, training_loader, data)
    # torch.save(traind_model.state_dict(), "saved_models/unet_encoder.pth")

    model.load_state_dict(torch.load("saved_models/unet_encoder.pth"))

    val = CustomDataset("image_d/test", transform=transforms.Compose(
        [transforms.ToTensor()]))


    model.eval()
    print(model)

    cam = GradCAM(model=model, target_layer=model.conv0[0], use_cuda=True)


    def segmentation_get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category, :, :].mean()
        return loss


    cam.get_loss = segmentation_get_loss
    target_category = 281

    num_image = 3
    image = val[num_image]
    image = image.to(device)
    image = image.unsqueeze(0)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image, target_category=1, aug_smooth=True)

    grayscale_cam = grayscale_cam[0, :]

    a = val[num_image].permute(1, 2, 0).numpy()

    visualization = show_cam_on_image(a, grayscale_cam)

    plt.imshow(visualization)
    plt.show()

    #
    # image = val[2]
    # image = image.to(device)
    # image = image.unsqueeze(0)
    # outputs = model(image)
    # image1 = outputs.cpu().clone()
    # image1 = image1.squeeze(0)
    # new_img_PIL = transforms.ToPILImage()(image1)
    # new_img_PIL.show()
    # summary(model, (3, 512, 512))