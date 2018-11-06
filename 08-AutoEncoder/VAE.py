import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import mnist2

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 2
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = mnist2.MNIST2('../data', transform=img_transform, download=False, do_file=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    print("in loss function")
    print("BCE = {}".format(BCE))
    print("recon_x = {}".format(recon_x))
    print("type x = {}".format(type(x)))
    print("x size = {}".format(x.size()))
    print("x = {}".format(x))
    print("mu = {}".format(mu))
    print("logvar = {}".format(logvar))
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_element).mul_(-0.5)
    mu2 = mu.pow(2)
    print("mu2 = {}".format(mu2))
    logvar2 = logvar.exp()
    print("logvar2 = {}".format(logvar2))
    KLD = mu2.add(logvar2)
    print("KLD = {}".format(KLD))
    KLD = KLD.mul(-1)
    print("KLD = {}".format(KLD))
    KLD = KLD.add(1)
    print("KLD = {}".format(KLD))
    KLD = KLD.add(logvar)
    print("KLD = {}".format(KLD))
    KLD = torch.sum(KLD)
    print("KLD = {}".format(KLD))
    KLD = KLD.mul(-0.5)
    print("KLD = {}".format(KLD))
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

for batch_idx, data in enumerate(dataloader):
    img, _ = data
    img = img.view(img.size(0), -1)
      
    recon_batch, mu, logvar = model(img)
    loss = loss_function(recon_batch, img, mu, logvar)

    save = to_img(recon_batch.cpu().data)
    save_image(save, './vae_img/image_{}.png'.format(1))
    break