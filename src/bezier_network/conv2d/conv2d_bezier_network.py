import numpy as np
import math
import torch.nn as nn
from bezier_network.bezier.bezier import ShapeBezierCurve
from bezier_network.conv2d.conv2d_linear_interpolation import Conv2dInterpolation

"""
File: conv2dBezierNetwork.py
Author: Jacob Vartuli-Schonberg
Email: jacob.vartuli.92@gmail.com
Github: https://github.com/OpenJ92
Description: 
"""

class Conv2dBezierNetwork(nn.Module):
    """
    Parameters
    ------------
    shape_in : np.array
    shape_out : np.array
    control_points : controlPoints
    bezier_samples : int
    layers : int
    bezier : bezierCurve
    network : list(torch.nn)
    callable : nn.Sequential
    """
    def __init__(self, shape_in, shape_out, control_points, bezier_samples, layers):
        super().__init__()
        self.bezier = ShapeBezierCurve(shape_in, shape_out, control_points)
        self.network = self.construct_Networks(layers, bezier_samples)
        self.callable_network = nn.Sequential(*self.construct_Sequential_Networks())

    def forward(self, A):
        return self.callable_network(A)

    def sample_bezier(self, num_samples):
        return self.bezier.sample(num_samples)

    def construct_Networks(self, layers_, samples):
        samples = self.sample_bezier(samples)
        network = []
        for i in range(samples.shape[0] - 1):
            linear_net = Conv2dInterpolation(samples[i], samples[i+1], layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self, reverse = False):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]

    def tensor_shape(self):
        tS = [self.network[0].sample_interpolation()]
        for i in range(1, len(self.network)):
            tS.append(self.network[i].sample_interpolation()[1:])
        tS = np.concatenate(tS, axis = 0)
        return tS

    def construct_voxel_diagram(self):
        tensor_shape = self.tensor_shape()
        max_h, max_w = np.max(tensor_shape[:,1]), np.max(tensor_shape[:,2])
        prisms = []
        for prism in tensor_shape:
            prism = np.array([4, *prism[1:]])
            prism_ = np.ones(shape = (prism))
            prism_[1:-1, 1:-1,1:-1] = np.zeros(shape = (prism - 2))
            diff_h, diff_w = max_h - prism[1], max_w - prism[2]

            uhp, lhp = [np.zeros(shape = (prism[0], int(func(diff_h / 2)), prism[2])) for func in [math.ceil, math.floor]] 
            uwp, lwp = [np.zeros(shape = (prism[0], prism[1] + diff_h, int(func(diff_w / 2)))) for func in [math.ceil, math.floor]]
            zp = np.zeros(shape = (25,max_h,max_w))
            prism_ = np.concatenate([uhp, prism_, lhp], axis = 1)
            prism_ = np.concatenate([uwp, prism_, lwp], axis = 2)
            prism_ = np.concatenate([zp, prism_, zp], axis = 0)
            
            prisms.append(prism_)
        return np.concatenate(prisms, axis = 0) == 1

    def plot(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        voxels = self.construct_voxel_diagram()
        fig = plt.figure(figsize = (18, 12))
        ax = fig.gca(projection='3d')
        ax.set_axis_off()
        ax.voxels(voxels, facecolors='red', edgecolor = 'k')

        plt.show()


conv2dbezierNetwork = Conv2dBezierNetwork
