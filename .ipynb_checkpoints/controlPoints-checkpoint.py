import numpy as np

class controlPointsUniformRandomEnclosingPrism():
    """
        Parameters
        -------------
        shape_in : np.array(shape = (dimension of tensor,))
            The desired shape of input tensor

        shape_out : np.array(shape = (dimension of tensor,))
            The desired shape of output tensor

    """
    def __init__(self, shape_in, shape_out):
        self.shape_in = shape_in
        self.shape_out = shape_out

    def __call__(self, sample_points):
        return self.construct_control_points(sample_points)

    def construct_control_points(self, sample_points):
        random_sample = np.random.random_sample(size=(self.shape_in.shape[0],sample_points))
        transform = np.diag(self.shape_out - self.shape_in)
        ones = np.ones(shape=(self.shape_out.shape[0],1))
        zeros = np.zeros(shape=(self.shape_in.shape[0],1))
        bezier_points = np.concatenate([zeros,
                                        random_sample,
                                        ones], axis=1)
        arg = np.argsort(np.apply_along_axis(np.linalg.norm,0,bezier_points))
        bezier_points = bezier_points[:,arg]
        bezier_points = (transform @ bezier_points) + self.shape_in.reshape(self.shape_in.shape[0],1)
        return bezier_points

class controlPointsVertebralWalk():
    """ 
        Parameters
        _____________
        shape_in : np.array(shape = (dimension of tensor,))
            The desired shape of input tensor

        shape_out : np.array(shape = (dimension of tensor,))
            The desired shape of output tensor

        spinous_process: np.array(shape = (dimension of space,))
            Vector that one will rotate and scale along t in [0,1]

        func : f(t) = (theta, radius)
            function that produces the scaling and rotation values 
            to apply to spinous_process vector along t
    """

    def __init__(self, shape_in, shape_out, spinous_process, func = lambda t: (t*np.pi/10,1/(t+1))):
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.spine_ = self.spine()
        self.spinous_process = spinous_process
        self.func = func

    def __call__(self, sample_points):
        return self.construct_control_points(sample_points)

    def spine(self):
        return self.shape_out - self.shape_in

    def construct_control_points(self, sample_points):
        control_points = []
        for i in range(sample_points + 1):
            base = self.shape_in
            spine_loc = base + (i/sample_points)*self.spine_
            theta, r = self.func(i)
            mat = self.rotate_Axis_Matrix(theta)
            spinous_process_current = mat @ self.make_unit(self.spinous_process)
            spinous_procsee_current_ = r*spinous_process_current
            control_points.append(spine_loc + spinous_process_current)
            print(r, theta, spine_loc)

        A = np.stack([self.shape_in] + control_points + [self.shape_out], axis = 1)
        return A

    def rotate_Axis_Matrix(self, theta):
        spine = self.make_unit(self.spine_)
        _11 = np.cos(theta) + (spine[0]**2)*(1 - np.cos(theta))
        _12 = spine[2]*spine[1]*(1 - np.cos(theta)) - spine[0]*np.sin(theta)
        _13 = spine[0]*spine[2]*(1 - np.cos(theta)) + spine[1]*np.sin(theta)
        _21 = spine[0]*spine[1]*(1 - np.cos(theta)) + spine[2]*np.sin(theta)
        _22 = np.cos(theta) + (spine[1]**2)*(1 - np.cos(theta))
        _23 = spine[1]*spine[2]*(1 - np.cos(theta)) - spine[0]*np.sin(theta)
        _31 = spine[2]*spine[0]*(1 - np.cos(theta)) - spine[1]*np.sin(theta)
        _32 = spine[2]*spine[1]*(1 - np.cos(theta)) + spine[0]*np.sin(theta)
        _33 = np.cos(theta) + (spine[2]**2)*(1 - np.cos(theta))
        Mat = np.array([[_11, _12, _13],
                        [_21, _22, _23],
                        [_31, _32, _33]])
        return Mat

    def make_unit(self, v):
        return v / np.linalg.norm(v)


if __name__ == '__main__':
    shape_in = np.array([10, 128, 128])
    shape_out = np.array([100, 4, 4])
    control2 = controlPointsUniformRandomEnclosingPrism(shape_in, shape_out)
    control1 = controlPointsVertebralWalk(shape_in, shape_out, np.array([1,1,1]))
