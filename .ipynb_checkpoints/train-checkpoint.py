import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from sklearn.naive_bayes import GaussianNB
import joblib
from dataLoad import NavBotData
from controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk
from conv2dBezierNetwork import conv2dbezierNetwork
from dataTransform import *
from pytorch_VariationalAutoEncoder import VAE
from bezierVAE import bezierVAE

class train():
    ## should the model class have the loss function defined on it? I think so.
    def __init__(self, data, model, optimizer, batch_size):
        pass

    def __call__(self):
        pass

if __name__ == '__main__':
    # Consider the following. Take each convolved and transformed element into an arbitrary latent space.
    # In that latent space, we have n 'attractors' corresponding to each class. We construct a regularization 
    # term which punishes for elements which do not resolve to regions near their corresponding attractors. 
    # My hypothesis is that unique elements to each class will have no problem migrating torwards these 
    # attractors, but elements which are shared among classes will be pulled torwards multiple attractors.
    # The effect is that when we pass a collection of n-grams through the trained network, unique elements 
    # will reside in a neighborhood of the attractors and shared behaviors will be relegated to the interior
    # of the system. Let's see if we can construct such a thing over the next few days.

    shape_init = [8, 8]

    shape_in_eC, shape_out_eC = np.array([10,*shape_init]), np.array([100, 2, 2])
    shape_in_eD, shape_out_eD = np.array([100*2*2]), np.array([25])
    shape_in_mv, shape_out_mv = shape_out_eD, np.array([15])

    control1 = controlPointsUniformRandomEnclosingPrism(shape_in_eC, shape_out_eC)(2)
    control2 = np.flip(control1, axis = 1)
    control4 = controlPointsUniformRandomEnclosingPrism(shape_in_eD, shape_out_eD)(2)
    control3 = np.flip(control4, axis = 1)
    control5 = controlPointsUniformRandomEnclosingPrism(shape_in_mv, shape_out_mv)(2)
    control6 = np.flip(control5, axis = 1)

    eC = {'shape_in': shape_in_eC, 'shape_out': shape_out_eC,
            'control_points': control1, 'bezier_samples': 5, 'layers': 0}
    eD = {'shape_in': shape_in_eD, 'shape_out': shape_out_eD,
            'control_points': control4, 'bezier_samples': 5, 'layers': 0}
    mv = {'shape_in': shape_in_mv, 'shape_out': np.array([3]),
            'control_points': control5, 'bezier_samples': 2, 'layers': 0}
    z = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control6, 'bezier_samples': 2, 'layers': 0}
    dD = {'shape_in': shape_out_mv, 'shape_out': shape_in_mv,
            'control_points': control3, 'bezier_samples': 5, 'layers': 0}
    dC = {'shape_in': shape_out_eD, 'shape_out': shape_in_eD,
            'control_points':  control2, 'bezier_samples': 5, 'layers': 0}

    vae = bezierVAE(eC, eD, mv, z, dD, dC)
    # vae = torch.load('bezierVAE.pt')
    # vae.train()

    XT = NavBotData('seriesID.npy', 'train_X/tensor_order_T', transforms.Compose([NGRAMDomain(8), MINMAX()]))
    y = NavBotData('seriesID.npy', 'train_y/standard', transform = transforms.Compose([labelEncoder('labelEncodery.joblib'), NGRAMRange(8, 128)]))

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
    
    epochs = 1000
    training_list = []
    for epoch in range(epochs):
        batch_indices = XT.batch_indices()
        batch = torch.from_numpy(XT[batch_indices]).float()

        reconstructed, z, mu, logvar = vae(batch)
        loss_ = vae.loss(reconstructed, batch, mu, logvar)

        training_list.append(np.array([epoch,loss_]))
        print(epoch, loss_/(8*8*10*batch.shape[0]))

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
