import numpy as np
import torch.nn as nn
from bezier_network.bezier.bezier import ShapeBezierCurve
from bezier_network.dense.dense_linear_interpolation import DenseInterpolation

class DenseBezierNetwork(nn.Module):

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
            linear_net = DenseInterpolation(samples[i], samples[i+1], layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]


densebezierNetwork = DenseBezierNetwork
