import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    pass

class UnFlatten(nn.Module):
    pass

class VAE(nn.Module):
    def __init__(self, image_channels=10, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(16,16,4), stride=1, padding=0, dilation=1, groups=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(8,8,2), stride=1, padding=0, dilation=1, groups=1, bias=True),
                nn.ReLU()
                )
        self.decoder = nn.Sequential()


if __name__ == '__main__':
    #### Important!! Lets consider how these convolutions change the shape of our tensor
    ####    This will become increasingly important as we begin to design the arch of the
    ####    system. Take a look at the following link on convolutional arithmetic:
    ####        https://arxiv.org/abs/1603.07285
    ####        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    # Messing around with Conv2d
    m = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 32, stride = 2)
    n = nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 16, stride = 2)
    o = nn.Conv2d(in_channels = 40, out_channels = 80, kernel_size = 8, stride = 2)

    x = nn.Linear(out_features=500, in_features=2000)
    y = nn.Linear(out_features=100, in_features=500)
    mu = nn.Linear(out_features=25, in_features=100)
    var = nn.Linear(out_features=25, in_features=100)


    A = torch.from_numpy(np.load('difftensor.npy')).float()[:,:,0:25]
    A = torch.einsum('ijkl -> lkij', [A])
    A = Variable(A)
    B = m(A)
    C = n(B)
    D = o(C)

    E = D.flatten(start_dim=1)

    V = x(E)
    G = y(V)

    _mu = mu(G)
    _var = var(G)

    ##      Also, we need to figure out how to use google cloud for this project! 
    ## https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/pytorch
    ## https://towardsdatascience.com/how-to-train-machine-learning-models-in-the-cloud-using-cloud-ml-engine-3f0d935294b3

    # Maybe a combination of Conv2d and Conv3d will work well. Remember, the only way to train this model will be through
    #   Cloud services. Be sure to learn how to use such services over then next few days.

    # messing around with Conv3d
    m = nn.Conv3d(in_channels=1,out_channels=2,kernel_size=(2,32,32),stride=1,dilation=2,padding=(0,0,0))
    n = nn.Conv3d(in_channels=2,out_channels=4,kernel_size=(2,16,16),stride=1,dilation=2,padding=(0,0,0))
    o = nn.Conv3d(in_channels=4,out_channels=8,kernel_size=(2,8,8),stride=1,dilation=2,padding=(0,0,0))
    p = nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(2,4,4),stride=1,dilation=2,padding=(0,0,0))
    w = nn.Conv3d(in_channels=16,out_channels=16,kernel_size=(2,4,4),stride=1,dilation=1,padding=(0,0,0))
    e = nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(1,10,10),stride=1,dilation=1,padding=(0,0,0))

    A_ = torch.from_numpy(np.load('difftensor.npy')).float()[:, :, :, 0:50]
    A_ = A.view(list(A.shape) + [1])
    A_ = torch.einsum('ijklm -> lmkij', [A])
    A_ = Variable(A)

    B_ = m(A_)
    C_ = n(B_)
    D_ = o(C_)
    E_ = p(D_)
    R_ = w(E_)
    T_ = e(R_)

    Y_ = T_.flatten(start_dim = 1)
    Z_ = nn.Linear(out_features=128,in_features=Y_.shape[1])(Y_)
    U_ = nn.Linear(out_features=64,in_features=128)(Z_)
    O_ = nn.Linear(out_features=32,in_features=64)(U_)
    mean_ = nn.Linear(out_features=16,in_features=32)(O_)
    logvar_ = nn.Linear(out_features=16,in_features=32)(O_)

