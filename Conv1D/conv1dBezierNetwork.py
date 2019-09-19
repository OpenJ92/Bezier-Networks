import numpy as np
import math
import torch
import torch.nn as nn
from BezierNetwork.Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk
from BezierNetwork.Bezier.Bezier import bezierCurve
from BezierNetwork.Conv2D.conv2dLinearInterpolation import Conv2dInterpolation

class conv1DbezierNetwork():
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
    # Remeber to reconstruct these files so that they're sutible for the 1 dimensional convolutional
    # network architecture. as it stands, these files should not compile. We should also consider 
    # construction of a testing suite using hypothesis.
    shape_in = np.array([10, 8, 64])
    shape_out = np.array([100, 64, 8])
    examples = 10
    sample_data = torch.from_numpy(np.random.random_sample(size=(examples, *shape_in))).float()
    sample_data_reverse = torch.from_numpy(np.random.random_sample(size=(examples, *shape_out))).float()
    c = Conv2dInterpolation(shape_in=shape_in, shape_out=shape_out, layers=4)
    d = Conv2dInterpolation(shape_in=shape_out, shape_out=shape_in, layers=4)
    _c = nn.Sequential(*c.construct_InterpolationNetwork())
    _d = nn.Sequential(*d.construct_InterpolationNetwork())

    A = _c(sample_data)
    B = _d(A)

    Q = _d(sample_data_reverse)
    W = _c(Q)
