import numpy as np
import torch.nn as nn
from bezier_network.bezier.bezier import ShapeBezierCurve

class Conv1dInterpolation:
    """
    Parameters
    --------------
    shape_in : np.array - Initial shape of tensor to be transformed.
    shape_out : np.array - Output shape of tensor to be tranformed to.
    control_points : np.array - Control Point set to control linear bezier form.
    function : bezierCurve - Bezier curve sampled for interior layers of interpolation.
    layers : int - number of layers in interpolation.

    Methods
    -------------
    sample_interpolatioon(self) - sample bezierCurve object
    contract(self, shape_in, shape_out) - construct a Conv2DTranspose operation 
                            given shape state in construct_InterpolationNetwork
    dialate(self, shape_in, shape_out) - construct a Conv2D operation given shape 
                            state in construct_InterpolationNetwork
    warp(self, shape_in, shape_out) - construct an itterated Conv2d to Conv2DTranspose
                            provided dimension delta are not all monotonic (increase/decrease)
    construct_InterpolationNetwork(self) - sample bezierCurve for self.layers points 
                            apply dialate, contract or warp where appropriate.

    """
    def __init__(self, shape_in, shape_out, layers):
        self.shape_in_ = np.asarray(shape_in, dtype=int)
        self.shape_out_ = np.asarray(shape_out, dtype=int)
        self.function = ShapeBezierCurve.linear(self.shape_in_, self.shape_out_)
        self.layers_ = layers
    
    def sample_interpolation(self):
        return self.function.sample(self.layers_ + 2)

    def construct_InterpolationNetwork(self):
        A = self.sample_interpolation()
        network = []
        for layer in range(self.layers_ + 1):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if np.all(shape_diff >= 0):
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm1d(num_features = A[layer+1][0]))
            elif np.all(shape_diff <= 0):
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm1d(num_features = A[layer+1][0]))
            else:
                j = self.warp(A[layer], A[layer+1])
                for i in j:
                    network.append(i)
        return network

    def contract(self, shape_in, shape_out):
        k_size = shape_in[1:]-shape_out[1:]+1
        return nn.Conv1d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=int(k_size[0]))

    def dialate(self, shape_in, shape_out):
        k_size = shape_out[1:]-shape_in[1:]+1
        return nn.ConvTranspose1d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=int(k_size[0]))

    def warp(self, shape_in, shape_out):
        shape_internal = np.array([
            shape_in[0] + np.absolute((shape_in[0] - shape_out[0]) // 2),
            shape_out[1],
        ])
        A = np.array([shape_in, shape_internal, shape_out])
        network = []
        for layer in range(2):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if np.all(shape_diff >= 0):
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm1d(num_features = A[layer+1][0]))
            elif np.all(shape_diff <= 0):
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm1d(num_features = A[layer+1][0]))
        return network 
