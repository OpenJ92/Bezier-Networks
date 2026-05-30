import numpy as np
import torch.nn as nn

from bezier_network.bezier.bezier import ShapeBezierCurve
from bezier_network.conv3d.conv3d_linear_interpolation import Conv3dInterpolation

class Conv3dBezierNetwork(nn.Module):
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
        self.layers = bezier_samples * layers

    def forward(self, A):
        return self.callable_network(A)

    def sample_bezier(self, num_samples):
        return self.bezier.sample(num_samples)

    def construct_Networks(self, layers_, samples):
        samples = self.sample_bezier(samples)
        network = []
        for i in range(samples.shape[0] - 1):
            linear_net = Conv3dInterpolation(samples[i], samples[i+1], layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self, reverse = False):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]

    def tensor_shape(self):
        tS = [self.network[0].sample_interpolation()]
        for i in range(1, len(self.network)):
            tS.append(self.network[i].sample_interpolation()[1:])
        return np.concatenate(tS, axis=0)

    def construct_voxel_diagram(self):
        pass

    def plot(self):
        pass


conv3dbezierNetwork = Conv3dBezierNetwork
