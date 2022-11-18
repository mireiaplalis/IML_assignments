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
""" Load the dataset
Dataset (Numpy npz file)
|- features (Numpy.ndarray)
|- labels (Numpy.ndarray)

The data is also provided in csv format.
"""

# TODO: all in one file
# TODO: Question 3

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

def run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, dataset, plot=True):
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
    n_features = 2 if dataset != 4 else 10
    base = Perceptron(alpha0=alpha0, max_iter=max_iter, n_features=n_features)
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
    plt.xlabel('Feature 1', fontsize=13)
    plt.ylabel('Feature 2', fontsize=13)
    plt.title("Decision regions for " + title,  fontsize=15, pad=15)
    plt.savefig("results/"+ title.replace(" ", "_") +".svg")
    plt.show()

def test_learning_rate(max_iter):
    datasets = [1, 2, 3, 4]
    learning_rates = np.logspace(-20, 10, 31)
    plt.close("all")
    plt.figure()
    for i, d in enumerate(datasets):
        scores = []
        for lr in learning_rates:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = f'../data/dataset{d}.npz'
            file_path = os.path.join(dir_path, file)
            train_X, train_y, test_X, test_y = prepare_data(file_path)
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
        for n in n_estimators:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = f'../data/dataset{d}.npz'
            file_path = os.path.join(dir_path, file)
            train_X, train_y, test_X, test_y = prepare_data(file_path)
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
        for s in sampling_percentages:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = f'../data/dataset{d}.npz'
            file_path = os.path.join(dir_path, file)
            train_X, train_y, test_X, test_y = prepare_data(file_path)
            score_i = [run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, max_iter, s, d, False) for _ in range(50)]
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = f'../data/dataset{d}.npz'
        file_path = os.path.join(dir_path, file)
        train_X, train_y, test_X, test_y = prepare_data(file_path)
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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4], default=1, required=False)
    parser.add_argument('--alpha0', type=float, default=1e-3, required=False)
    parser.add_argument('--max_iter', type=int, default=1000, required=False)
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

    # test_learning_rate(max_iter)

    # perceptron_score = run_base(train_X, train_y, test_X, test_y, max_iter, alpha0, dataset)
    # boosting_score = run_boosting(train_X, train_y, test_X, test_y, n_estimators, alpha0, 
    #                               max_iter, sampling_percentage, dataset)

    # accuracy_comparison(alpha0, max_iter, sampling_percentage)
    test_n_estimators(alpha0, max_iter, sampling_percentage)
    # test_sampling_percentage(alpha0, max_iter, n_estimators)