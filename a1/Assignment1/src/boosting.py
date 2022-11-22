from perceptron import Perceptron
import numpy as np


class BoostingAlgorithm:
    def __init__(self, n_estimators, alpha0, max_iter, sampling_percentage, sample_n, update="default"):
        """ Initialization of the parameters
        Args:
            n_estimators (int): number of base perceptron models
            Other parameters of your choice        
        """
        self.n_estimators = n_estimators
        self.alpha0 = alpha0
        self.max_iter = max_iter
        self.sampling_percentage = sampling_percentage
        self.sample_weights = np.full((sample_n,1), 1/sample_n)
        self.sample_n = sample_n
        self.estimators = []
        self.performances = np.array([])
        self.eps = 1e-10
        self.plot = False
        self.update = update

    def error(self, missclassified):
        error = np.sum(missclassified * self.sample_weights) / np.sum(self.sample_weights)
        return error

    def fit(self, X, y):
        """ Implementation of the training strategy
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
        """ 
        total_samples = y.size
        for _ in range(self.n_estimators):
            n_features = len(X[0]-1)
            base_i = Perceptron(alpha0=self.alpha0, max_iter=self.max_iter, n_features=n_features)
            sample_inds = np.random.choice(total_samples, int(total_samples * self.sampling_percentage),
                                       replace=True, p=self.sample_weights.flatten())
            if self.update == "restart":
                self.sample_weights = np.full((self.sample_n,1), 1/self.sample_n)
            train_X = X[sample_inds]
            train_y = y[sample_inds]
            base_i.fit(train_X, train_y)
            missclassified = (y.reshape(-1,1) != base_i.predict(X))
            error_i = self.error(missclassified)
            performance_i = 0.5 * np.log((1 - error_i + self.eps) / (error_i + self.eps))
            self.estimators.append(base_i)
            self.performances = np.append(self.performances, performance_i)
            if self.update == "only_increase":
                self.sample_weights *= np.exp(performance_i * missclassified)
            self.sample_weights *= np.exp(performance_i * (missclassified * 2 - 1))
            self.sample_weights /= np.sum(self.sample_weights)
        self.performances = np.array(self.performances)


    def predict(self, x):
        """ Implementation of the prediction strategy
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value, namely, class label(s)
        """
        predictions = np.array([base.predict(x).flatten() for base in self.estimators]).T
        if self.plot:
            final_pred = (np.sign(np.sum(predictions * self.performances, axis=1) / np.sum(self.performances)) + 1) / 2
        else:
            final_pred = np.sign(np.sum(predictions * self.performances, axis=1) / np.sum(self.performances))
        return final_pred.reshape(-1,1)