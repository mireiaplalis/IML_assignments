from perceptron import Perceptron
import numpy as np

class BoostingAlgorithm:
    # Implement your boosting algorithm here
    def __init__(self, n_estimators, alpha0, max_iter, sampling_percentage, sample_n, n_inputs, **kwargs):
        """ Initialize the parameters here 
        Args:
            n_estimators (int): number of base perceptron models
            Other parameters of your choice
        
        Think smartly on how to utilize multiple perceptron models
        """
        self.n_estimators = n_estimators
        self.alpha0 = alpha0
        self.max_iter = max_iter
        self.sampling_percentage = sampling_percentage
        self.sample_weights = np.full((sample_n,1), 1/sample_n)
        self.n_inputs = n_inputs
        self.estimators = []
        self.performances = []

    def error(self, missclassified):
        error = np.sum(missclassified * self.sample_weights) / np.sum(self.sample_weights)
        return error

    def fit(self, X, y, **kwargs):
        """ Implement the training strategy here
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
            Other parameters of your choice
        """ 
        total_samples = y.size
        for _ in range(self.n_estimators):
            base_i = Perceptron(alpha0=self.alpha0, max_iter=self.max_iter, n_inputs=self.n_inputs)
            sample_inds = np.random.choice(total_samples, int(total_samples * self.sampling_percentage),
                                       replace=True, p=self.sample_weights.flatten())
            train_X = X[sample_inds]
            train_y = y[sample_inds]
            base_i.fit(train_X, train_y)
            #import pdb; pdb.set_trace()
            missclassified = (y.reshape(-1,1) != base_i.predict(X))
            error_i = self.error(missclassified)
            if error_i != 0.0:
                performance_i = np.log((1 - error_i) / error_i)
            else:
                performance_i = 0
            self.estimators.append(base_i)
            self.performances.append(performance_i)
            self.sample_weights *= np.exp(performance_i * (missclassified * 2 - 1))
            # else:
            #     break
        self.performances = np.array(self.performances)


    def predict(self, x, **kwargs):
        """ Implement the prediction strategy here
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value, namely, class label(s)
        """
        # import pdb;
        # pdb.set_trace()
        predictions = np.array([base.predict(x).flatten() for base in self.estimators]).T
        final_pred = np.sign(np.sum(predictions * self.performances, axis=1) / np.sum(self.performances))
        return final_pred.reshape(-1,1)