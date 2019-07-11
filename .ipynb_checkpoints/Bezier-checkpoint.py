import numpy as np

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

    def __init__(self, shape_in, shape_out, sample_points):
        
        self.shape_in_ = shape_in
        self.shape_out_ = shape_out
        self.control_points = self.construct_control_points(sample_points - 2)
        
    def __call__(self, t):
        return np.ceil(self.evaluate(t)).astype('int')

    def construct_control_points(self, sample_points):
        random_sample=np.random.random_sample(size=(self.shape_in_.shape[0], sample_points))
        transform = np.diag(self.shape_out_ - self.shape_in_)
        ones = np.ones(shape=(self.shape_out_.shape[0], 1))
        zeros = np.zeros(shape=(self.shape_in_.shape[0], 1))
        bezier_points = np.concatenate([zeros,
                                        random_sample,
                                        ones],axis=1)
        arg = np.argsort(np.apply_along_axis(np.linalg.norm,0,bezier_points))
        bezier_points = bezier_points[:, arg]
        bezier_points = (transform @ bezier_points) + self.shape_in_.reshape(self.shape_in_.shape[0], 1)

        return bezier_points

    def evaluate(self, t):
        f = lambda t, c1, c2: (1-t)*c1 + t*c2

        temp_control = self.control_points
        while temp_control.shape[1] > 1:
            A = [f(t, temp_control[:, i], temp_control[:, i+1]) for i in range(temp_control.shape[1] - 1)]
            temp_control = np.stack(A, axis = 1)

        return temp_control

    

    
