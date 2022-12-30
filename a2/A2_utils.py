import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import itertools
import json
import matplotlib.patches as mpatches
import random
from sknn.ae import AutoEncoder

# THINGS TO EXPLAIN: outlier detection, scaling variables, removing features
# KMediods has scalability problems. DBSCAN does not work with fixed number of clusters
# We only plot a subset of the val data


def data_exploration(data):
    print(data.info())
    print(data.head())
    print("Occurrences of each class: \n", data['class'].value_counts())
    print("Presence of missing values: ", data.isnull().values.any())


def data_preprocessing(data_pd, int_labels=False, filter_outliers=False):
    data_pd = data_pd.drop(['field_ID', 'MJD', 'plate'], axis=1)

    if int_labels:
        data_pd = data_pd.replace('GALAXY', int(0))
        data_pd = data_pd.replace('QSO', int(1))
        data_pd = data_pd.replace('STAR', int(2))

    data_np = data_pd.to_numpy()
    X = data_np[:, :-1]
    y = data_np[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Scaling data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Outlier detection
    if filter_outliers:
        outlier_det = IsolationForest(n_estimators=10, warm_start=True)
        outlier_det.fit(X_train)
        X_train = X_train[np.where(outlier_det.predict(X_train) == 1)]
        y_train = y_train[np.where(outlier_det.predict(X_train) == 1)]


    return X_train, X_test, y_train, y_test

def data_preprocessing_for_cv(data_pd):
    data_pd = data_pd.drop(['field_ID', 'MJD', 'plate'], axis=1)
    data_pd = data_pd.replace('GALAXY', int(0))
    data_pd = data_pd.replace('QSO', int(1))
    data_pd = data_pd.replace('STAR', int(2))
    data_np = data_pd.to_numpy()
    X = data_np[:, :-1]
    y = data_np[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test