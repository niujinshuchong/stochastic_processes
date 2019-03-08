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
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--clip_gradient', type=bool, default=False, help='interval betwen image samples')
parser.add_argument('--results_dir', type=str, default='log', help='intermidia results of training')
parser.add_argument('--pretrain', type=str, default='', help='path to pretrained model')
parser.add_argument('--pretrain_classifier', type=str, default='', help='path to pretrained classifier')

opt = parser.parse_args()

os.makedirs(opt.results_dir, exist_ok=True)
os.makedirs(opt.results_dir+'/images', exist_ok=True)
os.makedirs(opt.results_dir+'/offsets', exist_ok=True)

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

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
            nn.Linear(256, opt.latent_dim),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)

        latent = self.model(img_flat)

        return latent


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim=int(np.prod(img_shape))):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, opt.latent_dim)
        )

    def forward(self, latent):
        noise = Variable(Tensor(np.random.normal(0, 1, latent.size()))).cuda()
        latent = self.model(torch.cat((latent, noise), dim=1))
        return latent


class MCMC(nn.Module):
    def __init__(self):
        super(MCMC, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.encoder = Encoder()
        self.transformer = Transformer()
        self.discriminator_latent = Discriminator(input_dim=opt.latent_dim)

        self.discriminator_latent_pair = Discriminator(input_dim=opt.latent_dim*2)

    def forward(self):
        pass
         
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

# Initialize generator and discriminator
mcmc = MCMC()

if cuda:
    mcmc.cuda()
    classifier.cuda()

classifier.eval()
mcmc.eval()
if opt.pretrain != '':
     pretrained_dict = torch.load(opt.pretrain)
     mcmc.load_state_dict(pretrained_dict)

if opt.pretrain_classifier != '':
    pretrained_dict = torch.load(opt.pretrain_classifier)
    classifier.load_state_dict(pretrained_dict)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)


# dataloader = torch.utils.data.DataLoader(
#     datasets.FashionMNIST('../../data/fashionmnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

samples = []
for i, (imgs, _) in enumerate(dataloader):
    # Configure input
    real_imgs = Variable(imgs.type(Tensor))

    # Generate a batch of images
    z = mcmc.encoder(real_imgs)

    for j in range(10000):
        z = mcmc.transformer(z)
        fake_imgs = mcmc.generator(z).detach()
        samples.append(fake_imgs)
        if len(samples) % 100 == 0:
            print(len(samples))
    break

samples = torch.cat(samples, dim=0)
pred_label = classifier(samples).argmax(dim=1)
print(pred_label.shape)
mat = np.zeros((10, 10))

for i in range(pred_label.size(0) -1):
    x, y = pred_label[i], pred_label[i+1]
    mat[x, y] += 1
print(mat)
mat = mat / mat.sum(axis=1).reshape(-1, 1)
print(mat)

#plt.figure(figsize=(16, 8))
#plt.subplot(121)
plt.figure()
ax = plt.gca()
im = ax.matshow(mat)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size='5%', pad=0.05)
plt.colorbar(im, cax=cax)

# plt.subplot(122)
# plt.hist(pred_label.data.cpu().numpy(), density=True)
plt.savefig('mat.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
pred_label = pred_label.data.cpu().numpy().astype(np.uint8)
unique, count = np.unique(pred_label, return_counts=True)
print(unique, count / pred_label.shape[0])
plt.hist(list(pred_label), weights=np.ones_like(pred_label) / float(pred_label.shape[0]))
plt.savefig('hist.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ----------
#  Training
# ----------

# ----------
#  sample
# ----------
# samples = []
# for i, (imgs, _) in enumerate(dataloader):
#     # Configure input
#     real_imgs = Variable(imgs.type(Tensor))
#
#     # Generate a batch of images
#     z = mcmc.encoder(real_imgs)
#     recons_imgs = mcmc.generator(z)
#     samples.extend([real_imgs, recons_imgs])
#     for j in range(25):
#         z = mcmc.transformer(z)
#         fake_imgs = mcmc.generator(z).detach()
#         samples.append(fake_imgs)
#
#     if (i+1) % 5 == 0:
#         samples = torch.cat(samples, dim=0)
#         save_image(samples,
#                    opt.results_dir + '/%d.png' % i, nrow=27, normalize=False)
#         samples = []
#         print(i)

