import sys
import os
import numpy as np
from perceptron import Perceptron
from boosting import BoostingAlgorithm

""" Load the dataset
Dataset (Numpy npz file)
|- features (Numpy.ndarray)
|- labels (Numpy.ndarray)

The data is also provided in csv format.
"""

def load_data(file_name='./dataset1.npz'):
    """ Load the Numpy npz format dataset 
    Args:
        file_name (string): name and path to the dataset (dataset1.npz, dataset2.npz, dataset3.npz)
    Returns:
        X (Numpy.ndarray): features
        y (Numpy.ndarray): 1D labels
    """
    import numpy as np
    data = np.load(file_name)
    X, y = data['features'], data['labels']
    return X, y


def run(**kwargs):
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
    pass

if __name__=='__main__':
    # TODO: Parse arguments. Dataset, learning rate, max iter...

    # Load dataset 1 by default
    X, y =load_data()
    print(X.shape)
    print(y.shape)
