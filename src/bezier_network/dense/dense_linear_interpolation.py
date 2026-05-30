import numpy as np
import torch.nn as nn
from bezier_network.bezier.bezier import ShapeBezierCurve

class DenseInterpolation:
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
        self.shape_in_ = np.asarray(shape_in, dtype=int)
        self.shape_out_ = np.asarray(shape_out, dtype=int)
        self.function = ShapeBezierCurve.linear(self.shape_in_, self.shape_out_)
        self.layers_ = layers

    def sample_interpolation(self):
        return self.function.sample(self.layers_ + 2).flatten()

    def construct_InterpolationNetwork(self):
        A = self.sample_interpolation()
        network = []
        for layer in range(self.layers_ + 1):
            network.append(nn.Linear(A[layer], A[layer+1]))
            network.append(nn.LeakyReLU())
            network.append(nn.BatchNorm1d(num_features = A[layer+1]))
        return network
