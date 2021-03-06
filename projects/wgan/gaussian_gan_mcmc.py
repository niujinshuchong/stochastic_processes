import argparse
import os
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random
from math import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=2, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

img_shape = (2,)

cuda = True if torch.cuda.is_available() else False

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
        )

        self.one_hot = nn.Linear(1024, 100)
        self.one_hot_offset = nn.Linear(100, int(np.prod(img_shape)))
        self.offset = nn.Linear(1024, int(np.prod(img_shape)))
        self.softmax = nn.Softmax()

    def forward(self, z):
        img = self.model(z)

        one_hot = self.one_hot(img)
        one_hot_offset = self.one_hot_offset(self.softmax(one_hot))
        #img = one_hot_offset + self.offset(img)
        img = self.offset(img)

        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
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
    def __init__(self, latent):
        super(Transformer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent)
        )

    def forward(self, latent):
        noise = Variable(Tensor(np.random.normal(0, 1, latent.size() ) ) ).cuda()
        latent = self.model(torch.cat((latent, noise), dim=1))
        return latent


class DiscriminatorT(nn.Module):
    def __init__(self, latent):
        super(DiscriminatorT, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, latent):
        validity = self.model(latent)
        return validity


def gaussian_mixture(batchsize, ndim, num_labels):
        if ndim % 2 != 0:
                raise Exception("ndim must be a multiple of 2.")

        def sample(x, y, label, num_labels):
                shift = 1.4
                r = 2.0 * np.pi / float(num_labels) * float(label)
                new_x = x * cos(r) - y * sin(r)
                new_y = x * sin(r) + y * cos(r)
                new_x += shift * cos(r)
                new_y += shift * sin(r)
                return np.array([new_x, new_y]).reshape((2,))

        x_var = 0.05
        y_var = 0.05
        x = np.random.normal(0, x_var, (batchsize, ndim // 2))
        y = np.random.normal(0, y_var, (batchsize, ndim // 2))
        z = np.empty((batchsize, ndim), dtype=np.float32)
        for batch in range(batchsize):
                for zi in range(ndim // 2):
                        z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
        return z


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

transformer = Transformer(opt.latent_dim)
discriminator_T = DiscriminatorT(opt.latent_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()

    transformer.cuda()
    discriminator_T.cuda()

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

optimizer_T = torch.optim.RMSprop(transformer.parameters(), lr=opt.lr)
optimizer_D_T = torch.optim.RMSprop(discriminator_T.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i in range(1000):

        imgs = gaussian_mixture(opt.batch_size, 2, 4)
        imgs = Tensor(imgs)
        #imgs = Tensor(np.random.uniform(low=1.3, high=5.7, size=(opt.batch_size, 2)))
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        optimizer_D_T.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        fake_z = transformer(z)

        # Generate a batch of images
        fake_imgs = generator(fake_z).detach()

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        loss_DT = -torch.mean(discriminator_T(z)) + torch.mean(discriminator_T(fake_z.detach()))

        loss_D.backward()
        loss_DT.backward()
        optimizer_D.step()
        optimizer_D_T.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            optimizer_T.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(fake_z)

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_T = -torch.mean(discriminator_T(fake_z))
            loss = loss_G + loss_T
            loss.backward()
            
            optimizer_G.step()
            optimizer_T.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs,
                                                            batches_done % 1000, 1000,
                                                            loss_D.item(), loss_G.item()))

        if batches_done % opt.sample_interval == 0:
            Y = gen_imgs.detach().cpu().numpy()
            plt.scatter(Y[:, 0], Y[:, 1], s=1)
            #gen_z = fake_z.detach().cpu().numpy()
            #plt.scatter(gen_z[:, 0], gen_z[:, 1])

            plt.savefig('tmp.png')
            plt.close()
            image = cv2.imread('tmp.png')
            cv2.imshow("image", image)
            cv2.waitKey(1)

        batches_done += 1

