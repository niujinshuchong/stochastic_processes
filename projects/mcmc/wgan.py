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
         

# Initialize generator and discriminator
mcmc = MCMC()

if cuda:
    mcmc.cuda()

if opt.pretrain != '':
     pretrained_dict = torch.load(opt.pretrain)
     mcmc.load_state_dict(pretrained_dict)


# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)


# Optimizers
optimizer_G = torch.optim.RMSprop(mcmc.generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(mcmc.discriminator.parameters(), lr=opt.lr)
optimizer_E = torch.optim.RMSprop(mcmc.encoder.parameters(), lr=opt.lr)
optimizer_T = torch.optim.RMSprop(mcmc.transformer.parameters(), lr=opt.lr)
optimizer_D_latent = torch.optim.RMSprop(mcmc.discriminator_latent.parameters(), lr=opt.lr)
optimizer_D_latent_pair = torch.optim.RMSprop(mcmc.discriminator_latent_pair.parameters(), lr=opt.lr)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        optimizer_D_latent.zero_grad()
        optimizer_D_latent_pair.zero_grad()

        # Generate a batch of images
        z = mcmc.encoder(real_imgs)
        transformed_z = mcmc.transformer(z)
        fake_imgs = mcmc.generator(torch.cat((z, transformed_z), dim=0)).detach()

        # Adversarial loss, we need to distinguish both fake images from z and transformed_z
        loss_D = -torch.mean(mcmc.discriminator(real_imgs)) + torch.mean(mcmc.discriminator(fake_imgs))

        loss_D.backward()
        if opt.clip_gradient:
            torch.nn.utils.clip_grad_value_(mcmc.discriminator.parameters(), 5)

        optimizer_D.step()

        # Adversarial loss of latent space
        loss_D_latent = -torch.mean(mcmc.discriminator_latent(z.detach())) + torch.mean(mcmc.discriminator_latent(transformed_z.detach()))

        loss_D_latent.backward()
        if opt.clip_gradient:
            torch.nn.utils.clip_grad_value_(mcmc.discriminator_latent.parameters(), 5)

        optimizer_D_latent.step()

        # Adversarial loss of transformer
        fake_z_pair = torch.cat((z, transformed_z), dim=1).detach()
        real_z_pair = torch.cat((z, z[torch.randperm(z.size(0))]), dim=1).detach()

        loss_D_latent_pair = -torch.mean(mcmc.discriminator_latent_pair(real_z_pair)) + torch.mean(mcmc.discriminator_latent_pair(fake_z_pair))

        loss_D_latent_pair.backward()
        if opt.clip_gradient:
            torch.nn.utils.clip_grad_value_(mcmc.discriminator_latent_pair.parameters(), 5)

        optimizer_D_latent_pair.step()


        # Clip weights of discriminator
        for p in mcmc.discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator and Encoder
            # -----------------

            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            optimizer_T.zero_grad()

            # Generate a batch of images
            z = mcmc.encoder(real_imgs)
            transformed_z = mcmc.transformer(z)

            gen_imgs = mcmc.generator(z)
            gen_imgs_transformed = mcmc.generator(transformed_z)

            # Adversarial loss
            loss_G = -torch.mean(mcmc.discriminator(torch.cat((gen_imgs, gen_imgs_transformed), dim=0)))

            # Reconstruction loss
            loss_BCE = F.binary_cross_entropy(gen_imgs, real_imgs)

            # Adversarial loss of latent space
            loss_T = -torch.mean(mcmc.discriminator_latent(transformed_z))

            # Adversarial loss of latent pair
            z_pair = torch.cat((z, transformed_z), dim=1)
            loss_T_pair = -torch.mean(mcmc.discriminator_latent_pair(z_pair))

            loss = loss_BCE + loss_G + 0.1*loss_T + 0.1*loss_T_pair

            loss.backward()

            # clip gradient
            if opt.clip_gradient:
                torch.nn.utils.clip_grad_value_(mcmc.encoder.parameters(), 5)
                torch.nn.utils.clip_grad_value_(mcmc.generator.parameters(), 5)
                torch.nn.utils.clip_grad_value_(mcmc.transformer.parameters(), 5)

            optimizer_G.step()
            optimizer_E.step()
            optimizer_T.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D: %f] [DT: %f] [DPair : %f] [G: %f] [BCE: %f] [T: %f] [TPair: %f]" %
                   (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader),
                   loss_D.item(), loss_D_latent.item(), loss_D_latent_pair.item(),
                    loss_G.item(), loss_BCE.item(), loss_T.item(), loss_T_pair.item()))

        if batches_done % opt.sample_interval == 0:
            save_image(torch.cat((gen_imgs.data[:100], gen_imgs_transformed.data[:100]), dim=0),
                       opt.results_dir + '/images/%d.png' % batches_done, nrow=20, normalize=False)

            z = z.detach().cpu().numpy()
            tz = transformed_z.detach().cpu().numpy()

            plt.plot([z[:, 0], tz[:, 0]], [z[:, 1], tz[:, 1]])
            plt.scatter(z[:,0], z[:, 1], s=1)
            plt.scatter(tz[:, 0], tz[:, 1], s=1)

            plt.savefig(opt.results_dir+'/offsets/%d.png'%batches_done)
            plt.close()
            image = cv2.imread(opt.results_dir+'/offsets/%d.png'%batches_done)
            cv2.imshow("image", image)
            cv2.waitKey(1)

        batches_done += 1

    torch.save(mcmc.state_dict(), opt.results_dir+'/mcmc_model.pkl')
