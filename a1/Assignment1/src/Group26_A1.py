import sys
import os
import numpy as np
from perceptron import Perceptron
from boosting import BoostingAlgorithm
import os 
import argparse
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import json

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

def load_data(file_name='../data/dataset1.npz'):
    """ Load the Numpy npz format dataset 
    Args:
        file_name (string): name and path to the dataset (dataset1.npz, dataset2.npz, dataset3.npz)
    Returns:
        X (Numpy.ndarray): features
        y (Numpy.ndarray): 1D labels
    """
    data = np.load(file_name)
    X, y = data['features'], data['labels']
    return X, y

def score_function(y_gth, y_pred):
    return 1 - np.sum(y_gth.reshape(-1,1) != y_pred) / y_gth.size

def prepare_data(dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = f'../data/dataset{dataset}.npz'
    file_path = os.path.join(dir_path, file)
    X, y =load_data(file_name=file_path)
    n = len(y)
    train_n = int(0.8 * n)
    perm_ind = np.random.permutation(n)
    train_X = X[perm_ind[:train_n]]
    train_y = y[perm_ind[:train_n]]
    test_X = X[perm_ind[train_n:]]
    test_y = y[perm_ind[train_n:]]
    return train_X, train_y, test_X, test_y

def run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, dataset, plot=True, init="random"):
    """ 
    Single run of the classifier
    """
    n_features = 2 if dataset != 4 else 10
    base = Perceptron(alpha0=alpha0, max_iter=max_iter, n_features=n_features, init=init)
    base.fit(train_X, train_y)
    y_pred = base.predict(test_X)
    score = score_function(test_y, y_pred)
    if plot:
        plot_regions(base, train_X, train_y, f"base learner on train data from dataset {dataset}")
        plot_regions(base, test_X, test_y, f"base learner on test data from dataset {dataset}")
        print("Base score = ", score)
    return score

def run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, max_iter, 
                sampling_percentage, dataset, plot=True):
    """ 
    Run of the Adaboost classifier
    """
    boosting = BoostingAlgorithm(n_estimators=n_estimators, alpha0=alpha0, max_iter=max_iter,
                                 sampling_percentage=sampling_percentage, sample_n=len(train_y))
    boosting.fit(train_X, train_y)
    y_pred = boosting.predict(test_X)
    score = score_function(test_y, y_pred)
    if plot:
        plot_regions(boosting, train_X, train_y, f"Adaboost on train data \n from dataset {dataset} with {n_estimators} estimators")
        plot_regions(boosting, test_X, test_y, f"Adaboost on test data \n from dataset {dataset} with {n_estimators} estimators")
        print("Boosting score = ", score)
    return score

def plot_regions(algorithm, data_X, data_y, title):
    # Plotting decision regions
    plt.figure(figsize=(7,6))
    algorithm.plot = True
    plot_decision_regions(data_X, ((data_y + 1) / 2).astype(int), clf=algorithm, legend=0, colors='blue,red, orange')
    algorithm.plot = False
    # Adding axes annotations
    plt.tick_params(axis='both', labelsize=15)
    plt.xlabel('Feature 1', fontsize=15)
    plt.ylabel('Feature 2', fontsize=15)
    plt.title(title.capitalize(),  fontsize=20, pad=10)
    plt.savefig("results/"+ title.replace(" ", "_") +".svg")
    plt.show()

def test_learning_rate(max_iter):
    datasets = [1, 2, 3, 4]
    learning_rates = np.logspace(-20, 10, 31)
    plt.close("all")
    plt.figure()
    for i, d in enumerate(datasets):
        scores = []
        train_X, train_y, test_X, test_y = prepare_data(d)
        for lr in learning_rates:
            score_i = [run_base(train_X, train_y, test_X, test_y, max_iter, lr, d, False) for _ in range(50)]
            scores.append(np.mean(score_i))
        plt.plot(np.array(learning_rates), np.array(scores), label=f"Dataset {d}")
    plt.xscale('log')
    plt.title("Accuracy per learning rate", fontsize=15, pad=15)
    plt.xlabel("Learning rate", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.legend(fontsize=11)
    plt.savefig("results/learning_rate_acc.svg")
    plt.show()

def test_n_estimators(alpha0, max_iter, sampling_percentage):
    datasets = [1, 2, 3, 4]
    n_estimators = np.logspace(0, 3, 10).astype(int)
    plt.close("all")
    plt.figure()
    for i, d in enumerate(datasets):
        scores = []
        train_X, train_y, test_X, test_y = prepare_data(d)
        for n in n_estimators:
            score_i = [run_boosting(train_X, train_y, test_X, test_y, n, alpha0, max_iter, sampling_percentage, d, False) for _ in range(50)]
            scores.append(np.mean(score_i))
        plt.plot(np.array(n_estimators), np.array(scores), label=f"Dataset {d}")
    plt.xscale('log')
    plt.title("Boosting accuracy per number of estimators", fontsize=15, pad=15)
    plt.xlabel("Number of estimators", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.legend(fontsize=11)
    plt.savefig("results/n_estimators_acc.svg")
    plt.show()    

def test_sampling_percentage(alpha0, max_iter, n_estimators):
    datasets = [1, 2, 3, 4]
    sampling_percentages = np.linspace(0, 1, 10)
    plt.close("all")
    plt.figure()
    for i, d in enumerate(datasets):
        scores = []
        train_X, train_y, test_X, test_y = prepare_data(d)
        for s in sampling_percentages:
            score_i = [run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, max_iter, s, d, False) for _ in range(5)]
            scores.append(np.mean(score_i))
        plt.plot(np.array(sampling_percentages), np.array(scores), label=f"Dataset {d}")
    plt.title("Boosting accuracy per sampling percentage", fontsize=15, pad=15)
    plt.xlabel("Sampling percentage", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.legend(fontsize=11)
    plt.savefig("results/samp_perc_acc.svg")
    plt.show()    

def accuracy_comparison(alpha0, max_iter, sampling_percentage):
    scores = {1: {}, 2: {}, 3: {}, 4: {}}
    for d in scores.keys():
        train_X, train_y, test_X, test_y = prepare_data(d)
        score_i_base = [run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, d, False) for _ in range(50)]
        scores[d]["base"] = np.mean(score_i_base)           
        n_estimators = [2, 5, 10, 100, 1000]
        scores[d]["boosting"] = {}
        for n in n_estimators:
            score_i_boost = [run_boosting(train_X, train_y, test_X, test_y, n, alpha0, max_iter, sampling_percentage, d, False) for _ in range(50)]
            scores[d]["boosting"][n] = np.mean(score_i_boost)
    with open(f"./results/performance_comparison.json", 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
        f.write('\n')   

def performance_per_iteration(alpha0, max_iter, n_estimators, sampling_percentage):
    datasets = [1, 2, 3, 4]
    update_methods = ["default", "restart", "only_increase"]
    fig, ax = plt.subplots(1, 4, figsize=(35, 10))
    for i, d in enumerate(datasets):
        train_X, train_y, test_X, test_y = prepare_data(d)
        for j, m in enumerate(update_methods):
            scores = []
            boosting = BoostingAlgorithm(n_estimators=1, alpha0=alpha0, max_iter=max_iter,
                                        sampling_percentage=sampling_percentage, sample_n=len(train_y))
            scores = []
            for _ in range(n_estimators):
                boosting.fit(train_X, train_y)
                y_pred = boosting.predict(test_X)
                scores.append(score_function(test_y, y_pred))
            ax[i].plot(np.arange(n_estimators), np.array(scores), label=f"Update method {j+1}")
        ax[i].set_xlabel("Iteration", fontsize=25)
        ax[i].set_ylabel("Accuracy", fontsize=25)
        ax[i].set_title(f"Dataset {d}", fontsize=30, pad=15)
        ax[i].tick_params(axis='both', labelsize=15)
        ax[i].legend(fontsize=25)
    fig.suptitle("Boosting accuracy per boosting iteration", fontsize=35, y=0.99)
    plt.savefig(f"results/iteration_acc.svg")
    plt.show()    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4], default=1, required=False)
    parser.add_argument('--alpha0', type=float, default=1e-2, required=False)
    parser.add_argument('--max_iter', type=int, default=1000, required=False)
    parser.add_argument('--n_estimators', type=int, default=10, required=False)
    parser.add_argument('--sampling_percentage', type=float, default=0.1, required=False)

    args = parser.parse_args()
    dataset = args.dataset
    alpha0 = args.alpha0
    max_iter = args.max_iter
    n_estimators = args.n_estimators
    sampling_percentage = args.sampling_percentage
    
    train_X, train_y, test_X, test_y = prepare_data(dataset)

    perceptron_score = run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, dataset)
    boosting_score = run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, 
                                  max_iter, sampling_percentage, dataset)

    test_learning_rate(max_iter)
    accuracy_comparison(alpha0, max_iter, sampling_percentage)
    test_n_estimators(alpha0, max_iter, sampling_percentage)
    test_sampling_percentage(alpha0, max_iter, n_estimators)
    performance_per_iteration(alpha0, max_iter, n_estimators, sampling_percentage)