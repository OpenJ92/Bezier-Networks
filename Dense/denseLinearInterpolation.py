import numpy as np
import torch
import torch.nn as nn
import BezierNetwork.Bezier.Bezier as Bezier

class DenseInterpolation():
    """
    Parameters
    -----------
    shape_in : np.array - Initial shape of tensor to be transformed.
    shape_out : np.array - Output shape of tensor to be tranformed to.
    control_points : np.array - Control Point set to control linear bezier form.
    function : bezierCurve - Bezier curve sampled for interior layers of interpolation.
    layers : int - number of layers in interpolation.

    Methods
    -----------
    sample_interpolatioon(self) - sample bezierCurve object
    construct_InterpolationNetwork(self) - sample bezierCurve for self.layers points apply, Linear.

    """
    def __init__(self, shape_in, shape_out, layers):
        self.shape_in_ = shape_in
        self.shape_out_ = shape_out
        self.control_points = np.stack([self.shape_in_, self.shape_out_], axis = 1)
        self.function = Bezier.bezierCurve(self.shape_in_, self.shape_out_, self.control_points)
        self.layers_ = layers

    def sample_interpolation(self):
        return np.rint([self.function(sample).flatten() for sample in np.linspace(0,1,self.layers_ + 2)]).astype('int').flatten()

    def construct_InterpolationNetwork(self):
        A = self.sample_interpolation()
        network = []
        for layer in range(self.layers_ + 1):
            network.append(nn.Linear(A[layer], A[layer+1]))
            network.append(nn.LeakyReLU())
            network.append(nn.BatchNorm1d(num_features = A[layer+1]))
        return network

if __name__ == '__main__':
    shape_in = np.array([1000])
    shape_out = np.array([10])
    dI = DenseInterpolation(shape_in, shape_out, 20)
