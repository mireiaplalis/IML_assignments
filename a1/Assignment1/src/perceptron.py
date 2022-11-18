import numpy as np

# TODO: try different initialization strategies

class Perceptron:
    # Implement your base learner here
    def __init__(self, alpha0, max_iter):
        """ Initialize the parameters here 
        Args:
            learning_rate (float or a collection of floats): your learning rate
            max_iter (int): the maximum number of training iterations
            Other parameters of your choice

        Examples ToDos:
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        Try different initialization strategies (as required in Question 2.3)
        """
        self.learning_rate = alpha0
        self.max_iter = max_iter
        self.alpha0 = alpha0
        self.weights =  np.random.randn(3,1)
        self.plot = False

    def fit(self, X, y, **kwargs):
        """ Implement the training strategy here
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
            Other parameters of your choice

        Example ToDos:
        # for _ in range(self.max_iter):
        #     Update the parameters of Perceptron according to the learning rate (self.learning_rate) and data (X, y)
        """ 
        ones = np.ones(X.shape[0]).reshape(-1,1)
        biased_X = np.concatenate([X, ones], axis=1)
        for _ in range(self.max_iter):
            output = np.matmul(biased_X, self.weights)
            pred = np.sign(output)
            self.weights -= self.learning_rate * np.matmul(biased_X.T, pred - y.reshape(-1,1))
            self.learning_rate -= self.alpha0 / self.max_iter
        return

    def predict(self, x) -> np.ndarray:
        """ Implement the prediction strategy here
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
        if self.plot:
            pred = (np.sign(np.matmul(biased_X, self.weights)) + 1) / 2
        else:
            pred = np.sign(np.matmul(biased_X, self.weights))
        return pred

