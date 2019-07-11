import numpy as np
import torch
import torch.nn as nn
import BezierNetwork.Bezier.Bezier as Bezier

class Conv3dInterpolation():
    def __init__(self, shape_in, shape_out, layers):
        self.shape_in_ = shape_in
        self.shape_out_ = shape_out
        self.control_points_ = np.stack([self.shape_in_, self.shape_out_], axis = 1)
        self.function = Bezier.bezierCurve(self.shape_in_, self.shape_out_, self.control_points_)
        self.layers_ = layers
    
    def sample_interpolation(self):
        return np.rint([self.function(sample).flatten() for sample in np.linspace(0,1,self.layers_ + 2)]).astype('int')

    def construct_InterpolationNetwork(self):
        A = self.sample_interpolation()
        network = []
        for layer in range(self.layers_ + 1):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if shape_diff[0] >= 0 and shape_diff[1] >= 0 and shape_diff[2] >= 0:
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm3d(num_features = A[layer+1][0]))
            elif shape_diff[0] <= 0 and shape_diff[1] <= 0 and shape_diff[2] <= 0:
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
                network.append(nn.BatchNorm3d(num_features = A[layer+1][0]))
            else:
                j = self.warp(A[layer], A[layer+1])
                for i in j:
                    network.append(i)
        return network

    def contract(self, shape_in, shape_out):
        k_size = shape_in[1:]-shape_out[1:]+1
        return nn.Conv3d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=k_size)

    def dialate(self, shape_in, shape_out):
        k_size = shape_out[1:]-shape_in[1:]+1
        return nn.ConvTranspose3d(out_channels=shape_out[0], in_channels=shape_in[0], kernel_size=k_size)

    def warp(self, shape_in, shape_out):
        shape_internal = np.array([shape_in[0] + np.absolute((shape_in[0] - shape_out[0]) // 3), shape_out[1], shape_in[2], shape_in[3]])
        shape_internal_1 = np.array([(shape_in[0] + 2*np.absolute((shape_in[0] - shape_out[0]) // 3)), shape_out[1], shape_out[2], shape_in[3]])
        s_i_1 = np.array([])
        A = np.array([shape_in, shape_internal, shape_internal_1, shape_out])
        network = []
        for layer in range(3):
            shape_diff = A[layer+1, 1:] - A[layer, 1:]
            if shape_diff[0] >= 0 and shape_diff[1] >= 0 and shape_diff[2] >= 0:
                network.append(self.dialate(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
            elif shape_diff[0] <= 0 and shape_diff[1] <= 0 and shape_diff[2] <= 0:
                network.append(self.contract(A[layer], A[layer+1]))
                network.append(nn.LeakyReLU())
        return network 

if __name__ == "__main__":
    samples = np.random.randint(2, 20, size = (50, 4))
    examples = 100

    for i in range(50):
        for j in range(50):
            shape_in, shape_out = samples[i], samples[j]
            sample_data = torch.from_numpy(np.random.random_sample(size=(examples, *shape_in))).float()
            print(f"shape_in = {shape_in}, shape_out = {shape_out}")
            try:
                c = Conv3dInterpolation(shape_in=shape_in, shape_out=shape_out, layers=4)
                _c = nn.Sequential(*c.construct_InterpolationNetwork())
                A = _c(sample_data)
            except Exception as e:
                print("\n")
                print(e, "\n")

