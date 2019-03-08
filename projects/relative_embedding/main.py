from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128*3, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 2)

        self.fc4 = nn.Linear(2, 20)
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z = self.fc3(x)
        return z

    def decode(self, z):
        z = self.relu(self.fc4(z))
        z = self.relu(self.fc5(z))
        return self.sigmoid(self.fc6(z))

    def forward(self, x):
        z= self.encode(x.view(-1, 784))
        x = self.decode(z)
        return x, z


latent_length = 2
model = VAE()
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, z):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # relative loss
    n = x.size(0)
    if n % 3 != 0: 
        #if True: 
        return BCE 
    x = x.view(-1, 784)
    ax, bx, cx = torch.chunk(x, chunks=3, dim=0)
    az, bz, cz = torch.chunk(z, chunks=3, dim=0)

    dx_ab = torch.sum(ax * bx, dim=1)
    dx_ac = torch.sum(ax * cx, dim=1)

    dz_ab = torch.sum(torch.pow(az - bz, 2), dim=1)
    dz_ac = torch.sum(torch.pow(az - cz, 2), dim=1)

    '''
    dz_ab = 1. / (1. + dz_ab)
    dz_ac = 1. / (1. + dz_ac)

    p = dz_ab / (dz_ab + dz_ac)

    mask = (dx_ab < dx_ac).type(torch.float32)
    relative_loss = - torch.mean(torch.log(mask * p + (1 - mask) * (1 - p)))
    return BCE + relative_loss
    '''

    distance = torch.mean(torch.pow(dx_ab / dx_ac - dz_ab / dz_ac, 2))
    return BCE + distance


optimizer = optim.Adam(model.parameters(), lr=1e-3)
#optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu = model(data)
        loss = loss_function(recon_batch, data, mu)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    mus, labels = [], []
    for i, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu = model(data)
        mus.append(mu.data.cpu().numpy())
        labels.append(label.numpy())
        test_loss += loss_function(recon_batch, data, mu).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    # show
    Y = np.concatenate(mus, axis=0)
    labels = np.concatenate(labels, axis=0)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels.reshape(-1))
    plt.savefig('tmp.png')
    plt.close()
    image = cv2.imread('tmp.png')
    cv2.imshow("image", image)
    cv2.waitKey(1)


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, latent_length))
    if args.cuda:
       sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')
    
