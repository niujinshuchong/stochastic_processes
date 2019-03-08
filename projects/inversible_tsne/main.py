import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt
from tsne import compute_P
import os

###############################
### network architecture ######
###############################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        feature = self.fc3(h2)
        return feature

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 28*28)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        recon_x = self.sigmoid(self.fc3(h2))
        return recon_x

class TSNE(nn.Module):
    def __init__(self):
        super(TSNE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        feature = self.encoder(x)
        recon_x = self.decoder(feature)
        return feature, recon_x


###############################
############ Loss #############
###############################
class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        
    def forward(self, feature, P):
        # construct joint distribution
        n, f = feature.size()
        sum_Y = torch.sum(torch.pow(feature, 2), dim=1, keepdim=True)  # (n, 1)
        num = -2. * torch.matmul(feature, feature.t())                   # (n, n)
        num = 1. / (1. + (sum_Y + num + sum_Y.t()))                    # (n, n)
        num = num * (1. - torch.eye(n).cuda())
        Q = num / torch.sum(num)
        Q = torch.clamp(Q, min=1e-12)

        # KL divergence between Q and P
        loss = P * torch.log(P / Q)
        return torch.sum(loss)


def show_features(Y, labels):
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig('tmp.png')
    plt.close()
    image = cv2.imread('tmp.png')
    cv2.imshow("image", image)
    cv2.waitKey(1)


###############################
########## training ###########
###############################
def main():
    # load data
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    print(X.shape, labels.shape)

    # precompute P
    if os.path.exists('mnist2500_p.npy'):
        P = np.load('mnist2500_p.npy')
    else:
        P = compute_P(X)
        np.save('mnist2500_p.npy', P)
    print(P.shape)

    # model
    t_sne = TSNE()
    t_sne.cuda() 

    # optimizer
    #optimizer = torch.optim.SGD(t_sne.parameters(), lr=1e-2, momentum=0.9)
    optimizer = torch.optim.Adam(t_sne.parameters(), lr=1e-3)

    input = torch.Tensor(X).cuda()
    P = torch.Tensor(P).cuda()

    # loss
    KL_loss = KLLoss()

    # training
    for i in range(10000):
        optimizer.zero_grad()
        feature, recon_x = t_sne(input)

        # loss
        bce_loss = F.binary_cross_entropy(recon_x, input)
        kl_loss = KL_loss(feature, P)
        loss = bce_loss + kl_loss

        # optimize
        loss.backward()
        optimizer.step()

        print("iter %04d: bce_loss: %.4lf KL_loss: %.4f Loss: %.4f"%(i, bce_loss.item(), kl_loss.item(), loss.item()))

        if (i+1) % 100 == 0:
            Y = feature.detach().cpu().numpy()
            show_features(Y, labels)

            r = recon_x.detach().cpu().view(-1, 1, 28, 28)[:64]
            save_image(r, 'results/sample_' + str(i) + '.png')

    #pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    #pylab.show()
 
if __name__ == "__main__":
    main()

