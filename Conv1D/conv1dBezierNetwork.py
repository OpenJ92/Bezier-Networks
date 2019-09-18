import numpy as np
import math
import torch
import torch.nn as nn
from BezierNetwork.Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk
from BezierNetwork.Bezier.Bezier import bezierCurve
from BezierNetwork.Conv2D.conv2dLinearInterpolation import Conv2dInterpolation

class conv1DbezierNetwork():
    def __init__(self, shape_in, shape_out, control_points, bezier_samples, layers):
        self.bezier = bezierCurve(shape_in, shape_out, control_points)
        self.network = self.construct_Networks(layers, bezier_samples)
        self.callable_network = nn.Sequential(*self.construct_Sequential_Networks())

    def __call__(self, A):
        return self.callable_network(A)

    def sample_bezier(self, num_samples):
        return np.apply_along_axis(lambda x: self.bezier(x), 1, np.linspace(0, 1, num_samples).reshape(num_samples, 1))
    def construct_Networks(self, layers_, samples):
        samples = self.sample_bezier(samples)
        network = []
        for i in range(samples.shape[0] - 1):
            linear_net = Conv2dInterpolation(samples[i,:,:].flatten(), samples[i+1,:,:].flatten(), layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self, reverse = False):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]

    def tensor_shape(self):
        for i in range(1, len(self.network)):
            tS.append(self.network[i].sample_interpolation()[1:])
        tS = np.concatenate(tS, axis = 0)
        return tS

if __name__ == "__main__":
    pass
