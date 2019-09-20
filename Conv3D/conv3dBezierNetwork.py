import numpy as np
import math
import torch
import torch.nn as nn
from BezierNetwork.Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism,controlPointsVertebralWalk
from BezierNetwork.Bezier.Bezier import bezierCurve
from BezierNetwork.Conv3D.conv3dLinearInterpolation import Conv3dInterpolation

"""
File: conv3dBezierNetwork.py
Author: Jacob Vartuli-Schonberg
Email: jacob.vartuli.92@gmail.com
Github: https://github.com/OpenJ92
Description: Bezier Convolutional 3D Neural Networks
"""

class conv3dbezierNetwork():
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
        self.layers = bezier_samples * layers

    def __call__(self, A):
        return self.callable_network(A)

    def sample_bezier(self, num_samples):
        return np.apply_along_axis(lambda x: self.bezier(x), 1, np.linspace(0, 1, num_samples).reshape(num_samples, 1))

    def construct_Networks(self, layers_, samples):
        samples = self.sample_bezier(samples)
        network = []
        for i in range(samples.shape[0] - 1):
            linear_net = Conv3dInterpolation(samples[i,:,:].flatten(), samples[i+1,:,:].flatten(), layers_)
            network.append(linear_net)
        return network

    def construct_Sequential_Networks(self, reverse = False):
        return [nn.Sequential(*net.construct_InterpolationNetwork()) for net in self.network]

    def tensor_shape(self):
        pass

    def coonstruct_voxel_diagram(self):
        pass

    def plot(self):
        pass

if __name__ == "__main__":
    samples = np.random.randint(2, 40, size = (50, 4))
    examples = 25 

    for i in range(50):
        for j in range(50):
            shape_in, shape_out = samples[i], samples[j]
            control_points = controlPointsUniformRandomEnclosingPrism(shape_in, shape_out)(2)
            bCN3D = conv3dbezierNetwork(shape_in, shape_out, control_points, 5, 2)
            sample_data = torch.from_numpy(np.random.random_sample(size=(examples, *shape_in))).float()
            print(f"shape_in = {shape_in}, shape_out = {shape_out}")
            try:
                Q = bCN3D(sample_data)
            except Exception as e:
                print("\n")
                print(e, "\n")
