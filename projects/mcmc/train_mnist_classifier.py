import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--clip_gradient', type=bool, default=False, help='interval betwen image samples')
parser.add_argument('--results_dir', type=str, default='log', help='interval betwen image samples')

opt = parser.parse_args()

os.makedirs(opt.results_dir, exist_ok=True)
os.makedirs(opt.results_dir+'/images', exist_ok=True)
os.makedirs(opt.results_dir+'/offsets', exist_ok=True)

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [ nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(img_shape)), 512, normalize=False),
            *block(512, 512),
            *block(512, 256),
            nn.Linear(256, 10),
            nn.Softmax()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)

        prob = self.model(img_flat)

        return prob


# Initialize generator and discriminator
classifier = Classifier()

if cuda:
    classifier.cuda()


# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=opt.batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# dataloader = torch.utils.data.DataLoader(
#     datasets.FashionMNIST('../../data/fashionmnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)


# Optimizers
optimizer = torch.optim.RMSprop(classifier.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    classifier.train()
    for i, (imgs, labels) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.cuda())

        optimizer.zero_grad()

        pred_prob = classifier(real_imgs)

        ce_loss = F.cross_entropy(pred_prob, labels.view(-1))

        ce_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]" %
                  (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader),
                   ce_loss.item()))

    classifier.eval()
    correct, total = 0., 0.
    for i, (imgs, labels) in enumerate(testloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.cuda()).view(-1)

        with torch.no_grad():
            pred_prob = classifier(real_imgs)

        pred = pred_prob.argmax(dim=1).view(-1)
        correct += np.sum(pred.data.cpu().numpy() == labels.data.cpu().numpy())
        total += pred.size(0)

    print("[Epoch %d] [ACC: %.4lf]" % (epoch, correct / total))

    torch.save(classifier.state_dict(), 'mnist_classifier.pkl')
