import numpy as np
import math
import torch
import torch.nn as nn
from controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk
from Bezier import bezierCurve
from denseLinearInterpolation import DenseInterpolation


class densebezierNetwork():

    def __init__(self, shape_in, shape_out, control_points, bezier_samples, layers):
        self.bezier = bezierCurve(shape_in, shape_out, control_points)
        self.network = self.construct_Networks(layers, bezier_samples)
        self.callable_network = nn.Sequential(*self.construct_Sequential_Networks())

    def __call__(self, A):
        return self.callable_network(A)

    def sample_bezier(self, num_samples):
        return np.apply_along_axis(lambda x: self.bezier(x), 1, np.linspace(0,1,num_samples).reshape(num_samples,1))

    def construct_Networks(self, layers_, samples):
        samples = self.sample_bezier(samples)
        network = []
        for i in range(samples.shape[0] - 1):
            linear_net = DenseInterpolation(samples[i], samples[i+1], layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]

if __name__ == '__main__':
    shape_in = np.array([1500])
    shape_out = np.array([10])
    control_points = controlPointsUniformRandomEnclosingPrism(shape_in, shape_out)(2)
    dBC = densebezierNetwork(shape_in, shape_out, control_points, 20, 3)

    data_sample = torch.from_numpy(np.random.random_sample(1500)).float()
    A = dBC(data_sample)
