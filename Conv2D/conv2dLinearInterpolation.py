import numpy as np
import torch
import torch.nn as nn
import BezierNetwork.Bezier.Bezier as Bezier

class Conv2dInterpolation():
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
        self.shape_in_ = shape_in
        self.shape_out_ = shape_out
        self.control_points = np.stack([self.shape_in_, self.shape_out_], axis = 1)
        self.function = Bezier.bezierCurve(self.shape_in_, self.shape_out_, self.control_points)
        self.layers_ = layers
    
    def sample_interpolation(self):
        return np.rint([self.function(sample).flatten() for sample in np.linspace(0,1,self.layers_ + 2)]).astype('int')

    def construct_InterpolationNetwork(self):
        A = self.sample_interpolation()
        network = []
        for layer in range(self.layers_ + 1):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if shape_diff[0] >= 0 and shape_diff[1] >= 0:
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm2d(num_features = A[layer+1][0]))
            elif shape_diff[0] <= 0 and shape_diff[1] <= 0:
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm2d(num_features = A[layer+1][0]))
            else:
                j = self.warp(A[layer], A[layer+1])
                for i in j:
                    network.append(i)
        return network

    def contract(self, shape_in, shape_out):
        k_size = shape_in[1:]-shape_out[1:]+1
        return nn.Conv2d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=k_size)

    def dialate(self, shape_in, shape_out):
        k_size = shape_out[1:]-shape_in[1:]+1
        return nn.ConvTranspose2d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=k_size)

    def warp(self, shape_in, shape_out):
        shape_internal = np.array([shape_in[0] + np.absolute((shape_in[0] - shape_out[0]) // 2), shape_out[1], shape_in[2]])
        A = np.array([shape_in, shape_internal, shape_out])
        network = []
        for layer in range(2):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if shape_diff[0] >= 0 and shape_diff[1] >= 0:
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm2d(num_features = A[layer+1][0]))
            elif shape_diff[0] <= 0 and shape_diff[1] <= 0:
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm2d(num_features = A[layer+1][0]))
        return network 

if __name__ == "__main__":
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
