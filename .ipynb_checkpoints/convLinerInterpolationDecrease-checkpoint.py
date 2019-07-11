import numpy as np
import torch
import torch.nn as nn
from Bezier import bezierCurve

class Conv2dInterpolation():
    """
    Paramters
    ---------
    
    shape_init: np.array([])
    shape_end: np.array([])
    layers: int
    function: <function>
    """
    def __init__(self, shape_init, shape_end, layers):
        self.shape_init_ = shape_in
        self.shape_end_ = shape_end
        self.function = bezierCurve(self.shape_in_, self.shape_out_, 2)
        self.layers_ = layers
    
    def sample_interpolation(self):
        return np.rint([self.function(sample) for i in np.linspace(0,1,self.layers)])
    
    def construct_InterpolationNetwork(self):
        pass
        

def construct_ConvTransposeNNSEQ(shape_init, shape_end, layers, function):
    # Paradigm:  Evaluate the function(monotonic decrease //for now) and build the kernel size from said function.
    #             Note the only parameter that's shanging is the kernel size
    
    if function == 'linear':
        tensor_shape = np.rint([shape_init + (shape_end - shape_init)*layer / layers for layer in range(layers + 1)]).astype('int')
    elif function == 'exponential':
        alpha = np.e ** (np.log(shape_end / shape_init) / layers)
        tensor_shape = np.rint([shape_init*(alpha**(layer)) for layer in range(layers + 1)]).astype('int')
    
    network = []
    for layer in range(layers):
        #print(tensor_shape[layer])
        network.append(nn.ConvTranspose2d(out_channels=tensor_shape[layer + 1][2],
                                          in_channels=tensor_shape[layer][2],
                                          kernel_size=tensor_shape[layer + 1][:2] - tensor_shape[layer][:2] + 1)
        )
        network.append(nn.LeakyReLU())
    return network, tensor_shape