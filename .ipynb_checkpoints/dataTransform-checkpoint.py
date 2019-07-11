import numpy as np

class MINMAX():
    def __call__(self, X):
        for i in range(X.shape[0]):
            min_ = X[i].min()
            max_ = X[i].max()
            X[i] = (X[i] - min_) / (max_ - min_)
        return X

class NGRAMDomain():
    def __init__(self, n):
        self.n = n

    def __call__(self, X):
        ngrams = []
        for i in range(X.shape[0]):
            for j in range(X.shape[2] - self.n):
                ngrams.append(X[i, :, j:j+self.n, j:j+self.n])
        return np.stack(ngrams, axis = 0)

class NGRAMRange():
    def __init__(self, n, domain_dim):
        self.n = n
        self.domain_dim = domain_dim

    def __call__(self, y):
        ngrams = []
        for i in range(y.shape[0]):
            for j in range(self.domain_dim - self.n):
                ngrams.append(y[i])
        import pdb; pdb.set_trace()
        return np.stack(ngrams, axis = 0)

class labelEncoder():
    def __init__(self, lePATH):
        import joblib
        self.le = joblib.load(lePATH)

    def __call__(self, y):
        return self.le.transform(y.flatten())


