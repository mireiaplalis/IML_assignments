import sys
import os
import numpy as np
from perceptron import Perceptron
from boosting import BoostingAlgorithm
import os 
import argparse
import matplotlib.pyplot as plt
from statistics import mean

""" Load the dataset
Dataset (Numpy npz file)
|- features (Numpy.ndarray)
|- labels (Numpy.ndarray)

The data is also provided in csv format.
"""

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

def prepare_data(filename):
    X, y =load_data(file_name=filename)
    n = len(y)
    train_n = int(0.8 * n)
    perm_ind = np.random.permutation(n)
    train_X = X[perm_ind[:train_n]]
    train_y = y[perm_ind[:train_n]]
    test_X = X[perm_ind[train_n:]]
    test_y = y[perm_ind[train_n:]]
    return train_X, train_y, test_X, test_y

def run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, n_experiments):
    """ Single run of your classifier
    # Load the data
    X, y = load_data()
    # Find a way to split the data into training and test sets
    -> X_train, y_train, X_test, y_test
    
    # Initialize the classifier
    base = Perceptron("your parameters")
    
    # Train the classifier
    base.fit(X_train, y_train, "other parameters")
   
    # Test and score the base learner using the test data
    y_pred = base.predict(X_test, "other parameters")
    score = SCORING(y_pred, y_test)
    """
    base_mean_lst = []
    learning_lst = []
    for _ in range(5):
        base_lst = []
        for _ in range(n_experiments):
            base = Perceptron(alpha0=alpha0, max_iter=max_iter, n_inputs=n_inputs)
            base.fit(train_X, train_y)
            y_pred = base.predict(test_X)
            score = score_function(test_y, y_pred)
            base_lst.append(score)
        base_mean_lst.append(mean(base_lst))
        learning_lst.append(alpha0)
        alpha0 *= 10
    #print("Base score = ", score)
    #print(learning_lst)
    return base_mean_lst, learning_lst

def run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, max_iter, 
                sampling_percentage, n_experiments):

    boosting_mean_lst = []
    const = 0.1
    percentage_lst = []
    for _ in range(10):
        boosting_lst = []
        for _ in range(n_experiments):
            boosting = BoostingAlgorithm(n_estimators=n_estimators, alpha0=alpha0, max_iter=max_iter,
                                         sampling_percentage=sampling_percentage, sample_n=len(train_y), n_inputs=n_inputs)
            boosting.fit(train_X, train_y)
            y_pred = boosting.predict(test_X)
            score = score_function(test_y, y_pred)
            boosting_lst.append(score)
        boosting_mean_lst.append(mean(boosting_lst))
        percentage_lst.append(sampling_percentage)
        sampling_percentage += const
    #print("Boosting score = ", score)
    return boosting_mean_lst, percentage_lst

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4], default=1, required=False)
    parser.add_argument('--alpha0', type=float, default=1e-3, required=False)
    parser.add_argument('--max_iter', type=int, default=100, required=False)
    parser.add_argument('--n_estimators', type=int, default=10, required=False)
    parser.add_argument('--sampling_percentage', type=float, default=0.8, required=False)
    parser.add_argument('--n_experiments', type=int, default=1, required=False)

    args = parser.parse_args()
    dataset = args.dataset
    alpha0 = args.alpha0
    max_iter = args.max_iter
    n_estimators = args.n_estimators
    sampling_percentage = args.sampling_percentage
    n_experiments = args.n_experiments
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = f'../data/dataset{dataset}.npz'
    file_path = os.path.join(dir_path, file)
    train_X, train_y, test_X, test_y = prepare_data(file_path)
    num_rows, num_cols = train_X.shape
    n_inputs = num_cols + 1

    perceptron_score, learning_lst = run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, n_experiments)
    boosting_score, percentage_lst = run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0,
                                  max_iter, sampling_percentage, n_experiments)

    #print(percentage_lst)
    #print(perceptron_score, learning_lst)
    learning_x = list(range(len(learning_lst)))
    plt.plot(learning_x, perceptron_score, label="Perceptron score")
    #plt.plot(boosting_score, label="Boosting score")
    plt.ylabel('Accuracy score')
    plt.xlabel('Learning rate')
    #plt.xticks(percentage_x, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xticks(learning_x, learning_lst)
    plt.legend()
    plt.show()
