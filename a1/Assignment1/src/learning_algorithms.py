import sys
import os
import numpy as np
from perceptron import Perceptron
from boosting import BoostingAlgorithm
import os 
import argparse
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

def run_base(train_X, train_y, test_X, test_y, max_iter, alpha0):
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
    base = Perceptron(alpha0=alpha0, max_iter=max_iter)
    base.fit(train_X, train_y)
    y_pred = base.predict(test_X)
    score = score_function(test_y, y_pred)
    print("Base score = ", score)
    return score

def run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, max_iter, 
                sampling_percentage):
    boosting = BoostingAlgorithm(n_estimators=n_estimators, alpha0=alpha0, max_iter=max_iter,
                                 sampling_percentage=sampling_percentage, sample_n=len(train_y))
    boosting.fit(train_X, train_y)
    y_pred = boosting.predict(test_X)
    score = score_function(test_y, y_pred)
    print("Boosting score = ", score)
    return score

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1, required=False)
    parser.add_argument('--alpha0', type=float, default=1e-3, required=False)
    parser.add_argument('--max_iter', type=int, default=100, required=False)
    parser.add_argument('--n_estimators', type=int, default=10, required=False)
    parser.add_argument('--sampling_percentage', type=float, default=0.8, required=False)

    args = parser.parse_args()
    dataset = args.dataset
    alpha0 = args.alpha0
    max_iter = args.max_iter
    n_estimators = args.n_estimators
    sampling_percentage = args.sampling_percentage
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = f'../data/dataset{dataset}.npz'
    file_path = os.path.join(dir_path, file)
    train_X, train_y, test_X, test_y = prepare_data(file_path)


    perceptron_score = run_base(train_X, train_y, test_X, test_y, max_iter, alpha0)
    boosting_score = run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, 
                                  max_iter, sampling_percentage)
