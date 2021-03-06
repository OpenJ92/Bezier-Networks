import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from Dense.denseBezierNetwork import densebezierNetwork
from Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk

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

class bezierVAEDense(nn.Module):
    def __init__(self, eD, mv, z, dD):
        super(bezierVAEDense, self).__init__()
        self.flatten = Flatten()
        self.encode_Dense = densebezierNetwork(**eD).callable_network
        self.mu_ = densebezierNetwork(**mv).callable_network
        self.var_ = densebezierNetwork(**mv).callable_network
        self.z_ = densebezierNetwork(**z).callable_network
        self.decode_Dense = densebezierNetwork(**dD).callable_network
        self.unflatten = UnFlatten()

    def bottleneck(self, h):
        mu, logvar = self.mu_(h), self.var_(h)
        std = torch.exp(.5*logvar)
        eps = torch.rand_like(std)
        z = mu + std*eps
        return z, mu, logvar

    def encode(self, x):
        flatten_ = self.flatten(x)
        eDense = self.encode_Dense(flatten_)
        z, mu, logvar = self.bottleneck(eDense)
        return z, mu, logvar

    def decode(self, z):
        Z = self.z_(z)
        dDense = self.decode_Dense(Z)
        sigdDense = nn.Sigmoid()(dDense)
        unflatten = self.unflatten(sigdDense, [10, 8, 8])
        return unflatten

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z, mu, logvar

    def loss(self, w, x, mu, logvar):
        BCE = F.binary_cross_entropy(w, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return KLD + BCE

if __name__ == '__main__':

    shape_in_eD, shape_out_eD = np.array([8*8*10]), np.array([100])
    shape_in_mv, shape_out_mv = shape_out_eD, np.array([15])

    control4 = controlPointsUniformRandomEnclosingPrism(shape_in_eD, shape_out_eD)(2)
    control5 = controlPointsUniformRandomEnclosingPrism(shape_in_mv, shape_out_mv)(2)
    control3 = np.flip(control4, axis = 1)
    control6 = np.flip(control5, axis = 1)
    
    eD = {'shape_in': shape_in_eD, 'shape_out': shape_out_eD,
            'control_points': control4, 'bezier_samples': 3, 'layers': 0}
    mv = {'shape_in': shape_in_mv, 'shape_out': np.array([15]),
            'control_points': control5, 'bezier_samples': 2, 'layers': 0}
    z = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control6, 'bezier_samples': 2, 'layers': 0}
    dD = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control3, 'bezier_samples': 3, 'layers': 0}

    vae = bezierVAEDense(eD, mv, z, dD)
    numpy_sample = np.random.random_sample(size = (480, 8*8*10))
    data = torch.from_numpy(numpy_sample).float()
    reconstructed, z, mu, logvar = vae(data)
    loss = vae.loss(reconstructed, data, mu, logvar)
