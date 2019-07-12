import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Dense.denseBezierNetwork import densebezierNetwork
from Conv2D.conv2dBezierNetwork import conv2dbezierNetwork
from Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk
from torchvision import transforms, utils
from dataTransform import *
from dataLoad import NavBotData

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    def __call__(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 80, 5, 5)
    def __call__(self, x, shape):
        return x.view(x.size(0), *shape)

class bezierVAE(nn.Module):
    def __init__(self, eC, eD, mv, z, dD, dC):
        super(bezierVAE, self).__init__()
        self.encoder_Conv = conv2dbezierNetwork(**eC).callable_network
        self.flatten = Flatten()
        self.encoder_Dense = densebezierNetwork(**eD).callable_network
        self.mu_ = densebezierNetwork(**mv).callable_network
        self.var_ = densebezierNetwork(**mv).callable_network
        self.z_ = densebezierNetwork(**z).callable_network
        self.decoder_Dense = densebezierNetwork(**dD).callable_network
        self.unflatten = UnFlatten()
        self.decoder_Conv = conv2dbezierNetwork(**dC).callable_network

    def bottleneck(self, h):
        mu, logvar = self.mu_(h), self.var_(h)
        std = torch.exp(.5*logvar)
        eps = torch.rand_like(std)
        z = mu + std*eps
        return z, mu, logvar

    def encode(self, x):
        eConv = self.encoder_Conv(x)
        flatten_ = self.flatten(eConv)
        eDense = self.encoder_Dense(flatten_)
        z, mu, logvar = self.bottleneck(eDense)
        return z, mu, logvar, eConv, flatten_, eDense

    def decode(self, z, eConv, flatten_, eDense):
        Z = self.z_(z)
        dDense = self.decoder_Dense(Z + eDense)
        unflatten_ = self.unflatten(dDense,[100,2,2])
        dConv = self.decoder_Conv(unflatten_ + eConv)
        sigdConv = nn.Sigmoid()(dConv)
        return sigdConv

    def forward(self, x):
        z, mu, logvar, eConv, flatten_, eDense = self.encode(x)
        reconstructed = self.decode(z,eConv, flatten_, eDense)
        return reconstructed, z, mu, logvar

    def loss(self, w, x, mu, logvar):
        BCE = F.binary_cross_entropy(w, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD

if __name__ == '__main__':

    shape_init = [32, 32]

    shape_in_eC, shape_out_eC = np.array([10,*shape_init]), np.array([100, 2, 2])
    shape_in_eD, shape_out_eD = np.array([100*2*2]), np.array([25])
    shape_in_mv, shape_out_mv = shape_out_eD, np.array([3])

    control1 = controlPointsUniformRandomEnclosingPrism(shape_in_eC, shape_out_eC)(2)
    control2 = np.flip(control1, axis = 1)
    control4 = controlPointsUniformRandomEnclosingPrism(shape_in_eD, shape_out_eD)(2)
    control3 = np.flip(control4, axis = 1)
    control5 = controlPointsUniformRandomEnclosingPrism(shape_in_mv, shape_out_mv)(2)
    control6 = np.flip(control5, axis = 1)

    eC = {'shape_in': shape_in_eC, 'shape_out': shape_out_eC,
            'control_points': control1, 'bezier_samples': 20, 'layers': 2}
    eD = {'shape_in': shape_in_eD, 'shape_out': shape_out_eD,
            'control_points': control4, 'bezier_samples': 20, 'layers': 2}
    mv = {'shape_in': shape_in_mv, 'shape_out': np.array([3]),
            'control_points': control5, 'bezier_samples': 20, 'layers': 2}
    z = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control6, 'bezier_samples': 20, 'layers': 2}
    dD = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control3, 'bezier_samples': 20, 'layers': 2}
    dC = {'shape_in': shape_out_eD, 'shape_out': shape_in_eD,
            'control_points':  control2, 'bezier_samples': 20, 'layers': 2}

    bVAE = bezierVAE(eC, eD, mv, z, dD, dC)
    numpy_sample = np.random.random_sample(size = (480, 10, 32, 32))
    data = torch.from_numpy(numpy_sample).float()
    reconstructed, z, mu, logvar = bVAE(data)
    loss = bVAE.loss(reconstructed, data, mu, logvar)
