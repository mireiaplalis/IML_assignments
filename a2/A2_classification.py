import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import json
import random
from sknn.ae import AutoEncoder
from A2_utils import data_exploration, data_preprocessing, data_preprocessing_for_cv
from A2_clustering import dimensionality_reduction_fit_transform
from decision_regions_custom import plot_decision_regions
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def get_cv_scores(classifiers, X, y):
    k_folds = KFold(n_splits = 5)
    cv_scores = {}
    for clf in classifiers:
        pipeline = make_pipeline(StandardScaler(), clf)
        cv_scores[clf.__class__.__name__] = cross_val_score(pipeline, X, y, cv=k_folds).mean()
    return cv_scores


def visualize_hp_influence(model, X, y, default_hp, studied_hp, study_range, discrete=True):
    """
    Generates a plot showing the influence of a parameter in the cross validation
    score.
    Inputs:
    
    Outputs:
    - plot: line plot with the cv score for different values of the parameter.
    """
    plt.close("all")
    plt.figure()
    k_folds = KFold(n_splits = 5)
    cv_scores_mean = []
    cv_scores_max = []
    cv_scores_min = []
    for hp_value in study_range:
        hp_config = default_hp.copy()
        if discrete:
            hp_value = int(hp_value)
        hp_config[studied_hp] = hp_value
        model_instance = model(**hp_config)
        pipeline = make_pipeline(StandardScaler(), model_instance)
        cv_fold_scores = cross_val_score(pipeline, X, y, cv=k_folds)
        cv_scores_mean.append(cv_fold_scores.mean())
        cv_scores_max.append(cv_fold_scores.max())
        cv_scores_min.append(cv_fold_scores.min())

    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_min = np.array(cv_scores_min)
    cv_scores_max = np.array(cv_scores_max)
    error_bars = np.stack((cv_scores_mean - cv_scores_min, cv_scores_max- cv_scores_mean))
    plt.errorbar(study_range, cv_scores_mean, yerr=error_bars)
    plt.title(f"Influence of {studied_hp} \n in performance of {model.__name__}", fontsize=15, pad=10)
    plt.ylabel("CV score", fontsize=13)
    plt.xlabel(studied_hp, fontsize=13)
    plt.legend(fontsize=12)
    plt.savefig("./results/performance_" + studied_hp.replace(" ", "_") + ".svg")
    plt.show() 

def study_nestimators_RF(X, y):
    default_hp = {}
    studied_hp = "n_estimators"
    study_range = np.logspace(1, 3, 10)
    model = RandomForestClassifier
    visualize_hp_influence(model, X, y, default_hp, studied_hp, study_range)

def plot_regions(algorithm, dim_red, data_X, data_y, title):
    # Plotting decision regions
    fig = plt.figure(figsize=(7,6))
    algorithm.plot = True
    plot_decision_regions(data_X[:300], data_y[:300], clf=algorithm, dim_red=dim_red, legend=0, colors='blue,red,orange')
    algorithm.plot = False
    # Adding axes annotations
    plt.xlabel('Feature 1', fontsize=17)
    plt.ylabel('Feature 2', fontsize=17)
    plt.title("Decision regions for \n" + title,  fontsize=25, pad=15)
    fig_title = fig.axes[0].title
    plt.savefig("results/decisior_reg_"+ title.replace(" ", "_") +".svg", bbox_extra_artists=(fig_title,), bbox_inches='tight')
    plt.show()


def decision_regions(X_train, y_train, X_test, y_test, classifiers):
    pca, x_train_pca = dimensionality_reduction_fit_transform(PCA, 2, X_train)
    X_test_pca = pca.transform(X_test)

    for clf in classifiers:
        print(clf.__class__.__name__)
        clf.fit(X_train, y_train)
        plot_regions(clf, pca, pca.transform(X_test), y_test.astype(np.int_), clf.__class__.__name__)


if __name__ == '__main__':
    data = pd.read_csv("A2_data.csv")
    data_exploration(data)    

    np.random.seed(0)
    random.seed(0)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv  = data_preprocessing_for_cv(data)
    X_train, X_test, y_train, y_test = data_preprocessing(data, int_labels=True)

    # study_nestimators_RF(X_train_cv, y_train_cv)

    log_reg = LogisticRegression(solver='lbfgs', C=10**10, multi_class='multinomial', class_weight='balanced')
    random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    svc_linear = LinearSVC(C=float(1.0), class_weight='balanced')
    mlp = MLPClassifier(hidden_layer_sizes=(200,100,50,), random_state=1, max_iter=300)
    dtc = DecisionTreeClassifier(class_weight='balanced')

    classifiers = [log_reg, svc_linear, mlp, dtc, random_forest]
    decision_regions(X_train, y_train, X_test, y_test, classifiers)

    cv_scores = get_cv_scores(classifiers, X_train_cv, y_train_cv)

    with open(f"./results/classification_cv_scores.json", 'w') as f:
        json.dump(cv_scores, f, sort_keys=True, indent=4)
        f.write('\n')


    random_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    rf_test_acc = accuracy_score(y_test, y_pred)
    print("Generalization score of the random forest: ", rf_test_acc)

