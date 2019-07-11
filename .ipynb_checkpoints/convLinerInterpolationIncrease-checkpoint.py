import numpy as np
import torch
import torch.nn as nn

def construct_ConvNNSEQ(shape_init, shape_end, layers, function):
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
        network.append(nn.Conv2d(out_channels=tensor_shape[layer + 1][2], 
                                 in_channels=tensor_shape[layer][2], 
                                 kernel_size=tensor_shape[layer][:2] - tensor_shape[layer + 1][:2] + 1
                                )
                            )
        network.append(nn.LeakyReLU())
    return network, tensor_shape
