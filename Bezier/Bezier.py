import numpy as np
from bezier_network.Bezier.controlPoints import controlPointsUniformRandomEnclosingPrism, controlPointsVertebralWalk

class bezierCurve():
    """
        Parameters
        ------------
        shape_in : np.array(shape = (dimension of tensor,))
            The desired shape of input tensor

        shape_out : np.array(shape = (dimesion of tensor,))
            The desired shape of output tensor

        sample_points : np.array(shape = (dimension of tensor, number of internal samples))
            Set of ordered control points for generation of Bezeir Curve.
                """

    def __init__(self, shape_in, shape_out, control_points):
        self.shape_in_ = shape_in
        self.shape_out_ = shape_out
        self.control_points = control_points
        
    def __call__(self, t):
        return np.ceil(self.evaluate(t)).astype('int')

    def evaluate(self, t):
        f = lambda t, c1, c2: (1-t)*c1 + t*c2
        temp_control = self.control_points
        while temp_control.shape[1] > 1:
            A = [f(t, temp_control[:, i], temp_control[:, i+1]) for i in range(temp_control.shape[1] - 1)]
            temp_control = np.stack(A, axis = 1)

        return temp_control

if __name__ == '__main__':
    shape_in = np.array([10,128,128])
    shape_out = np.array([100,4,4])
    control_points = controlPointsUniformRandomEnclosingPrism(shape_in, shape_out)(10)
    control_points = controlPointsVertebralWalk(shape_in, shape_out, np.array([1,1,1]))(10)
    bezier = bezierCurve(shape_in,shape_out, control_points)

