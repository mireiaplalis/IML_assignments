import numpy as np

class Perceptron:
    def __init__(self, alpha0, max_iter, n_features, init="random"):
        """ Initialization of the parameters
        Args:
            learning_rate (float or a collection of floats): your learning rate
            max_iter (int): the maximum number of training iterations
            Other parameters of your choice
        """
        self.learning_rate = alpha0
        self.max_iter = max_iter
        self.alpha0 = alpha0
        self.weights =  np.random.randn(n_features + 1,1) if init == "random" else np.zeros((n_features + 1, 1))
        self.plot = False

    def fit(self, X, y):
        """ Implementation of the training strategy
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
        """ 
        ones = np.ones(X.shape[0]).reshape(-1,1)
        biased_X = np.concatenate([X, ones], axis=1)
        for _ in range(self.max_iter):
            output = np.matmul(biased_X, self.weights)
            pred = np.sign(output) + (output == 0)
            self.weights -= self.learning_rate * np.matmul(biased_X.T, pred - y.reshape(-1,1))
            self.learning_rate -= self.alpha0 / self.max_iter
        return

    def predict(self, x) -> np.ndarray:
        """ Implementation of the prediction strategy
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value(s), namely, class label(s), others of your choice
        """ 
        if len(x.shape) < 2:
            x = x.reshape(1,-1)
        ones = np.ones(x.shape[0]).reshape(-1,1)
        biased_X = np.concatenate([x, ones], axis=1)
        output = np.matmul(biased_X, self.weights)
        if self.plot:
            pred = (np.sign(output) + (output == 0) + 1) / 2
        else:
            pred = np.sign(output) + (output == 0)
        return pred

