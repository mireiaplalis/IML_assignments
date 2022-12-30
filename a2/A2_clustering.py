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
from A2_utils import data_exploration, data_preprocessing

# THINGS TO EXPLAIN: outlier detection, scaling variables, removing features
# KMediods has scalability problems. DBSCAN does not work with fixed number of clusters
# We only plot a subset of the val data

# TODO: autoencoder

# **************************************************#
############# Dimensionality reduction ##############
# **************************************************#

def dimensionality_reduction_fit_transform(method, components, x):
    """
    Applies the specified dimensionality reduction method to the data in x

    :param method: (callable) dimensionality reduction method to be used
    :components: (int) number of dimensions to which we want to reduce
    :param x: (ndarray) data
    :return: (ndarray) data with reduced dimension
    """ 
    if(method == AutoEncoder):
        method_instance = AutoEncoder(
            layers=[
                ("Sigmoid", 8),
                ("Linear", )
            ],
            learning_rate=0.01,
            n_iter=25
        )
    else:
        method_instance = method(n_components=components)
    x_dim_red = method_instance.fit_transform(x)
    return method_instance, x_dim_red

def dimensionality_reduction_fit(method, components, x):
    """
    Applies the specified dimensionality reduction method to the data in x

    :param method: (callable) dimensionality reduction method to be used
    :components: (int) number of dimensions to which we want to reduce
    :param x: (ndarray) data
    :return: (ndarray) data with reduced dimension
    """ 
    if(method == AutoEncoder):
        method_instance = AutoEncoder(
            layers=[
                ("Sigmoid", 8),
                ("Linear", )
            ],
            learning_rate=0.01,
            n_iter=25
        )
    else:
        method_instance = method(n_components=components)
    method_instance.fit(x)
    return method_instance

def clustering_fit_predict(method, X_train, X_test, k=3):
    method_instance = method(n_clusters=k).fit(X_train)
    clusters = method_instance.predict(X_test)
    return clusters

def add_points_to_scatter_plot(ax, x, indices, color, marker, title):
    """
    Adds the points specified in indices to the scatter plot in ax

    :param ax: scatter plot
    :param x: (ndarray(N, 256)) data
    :param indices: (ndarray(n,)) indices to select the data to be plotted
    :param color: (ndarray(4,)) RGBA color for the points
    """ 
    x_i = x[indices, :]
    ax.scatter(x_i[:,0], x_i[:,1], color=color, marker=marker)
    ax.set_xlabel(title.lower() + "_1", fontsize=30)
    ax.set_ylabel(title.lower() + "_2", fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    ax.set_title(title.upper(), fontsize=35, pad=10)
    return

def get_accuracy_clustering(y_test, y_cluster):
    perms = list(itertools.permutations(range(3)))
    accuracies = []
    labels = {"GALAXY": 0, "QSO": 1, "STAR": 2}
    y_test_int = np.array([labels[i] for i in y_test])
    for p in perms:
        y_cluster_p = np.array([p[i] for i in y_cluster])
        accuracies.append(np.sum(y_cluster_p == y_test_int) / len(y_test))
    return np.max(accuracies), perms[np.argmax(accuracies)]

def dimensionality_reduction(X_train, y_train, X_test, y_test, name, clustering, red_before=False, verbose = False):
    """
    Applies three different dimensionality reduction methods to the data
    """
    pca, x_train_pca = dimensionality_reduction_fit_transform(PCA, 2, X_train)
    umap_ins, x_train_umap = dimensionality_reduction_fit_transform(umap.UMAP, 2, X_train) #(umap.UMAP, 2, x)
    ae, x_train_autoencoder = dimensionality_reduction_fit_transform(AutoEncoder, 2, X_train) #(TSNE, 2, x)

    X_test_pca = pca.transform(X_test)
    X_test_umap = umap_ins.transform(X_test)
    X_test_autoencoder = ae.transform(X_test)

    tab10 = cm.get_cmap('tab10')
    labels = np.unique(y_test)
    colors = tab10(range(len(labels)))
    fig, ax_list = plt.subplots(1,3,figsize=(35, 11))
    # fig.tight_layout(pad=10)

    clusters = np.arange(3)
    results = {}
    markers = ["o", "^", "s"] 

    methods = ["pca", "autoencoder", "umap"]

    import pdb; pdb.set_trace()
    print(np.sum(X_test_pca - X_test_umap >= 1e-8))
    print(np.sum(X_test_pca - X_test_autoencoder >= 1e-8))


    for m, ax in enumerate(ax_list):
        x_train_red = locals()["x_train_" + methods[m]]
        X_test_red = locals()["X_test_" + methods[m]]
        if red_before:
            y_cluster = clustering_fit_predict(clustering, x_train_red, X_test_red)
        else:
            y_cluster = clustering_fit_predict(clustering, X_train, X_test)
        results[methods[m]], perm = get_accuracy_clustering(y_test, y_cluster)
        y_cluster_p = np.array([perm[i] for i in y_cluster])
        for j, marker in zip(clusters, markers):
            for i, color in zip(labels, colors):
                indices = np.where((y_test[:1000] == i) & (y_cluster_p[:1000] == j))[0]
                add_points_to_scatter_plot(ax, X_test_red, indices, color, marker, methods[m])
    patches = [mpatches.Patch(color=color, label=label) for color,label in zip(colors, labels)]
    legend = fig.legend(labels, ncol=4, prop={'size': 30}, title="True label", handles=patches, loc='upper center', bbox_to_anchor=(0.5,-0.002))
    legend.get_title().set_fontsize('30')
    title = "Clustering " + name + " dimensionality reduction"
    fig_title = fig.suptitle(title, fontsize = 45, y=1)
    fig.savefig("results/" + title.replace(" ", "_")+ ".svg", bbox_extra_artists=(legend,fig_title), bbox_inches='tight')
    if verbose:
        plt.show()
        plt.close()
    return results

if __name__ == '__main__':
    data = pd.read_csv("A2_data.csv")
    np.random.seed(0)
    random.seed(0)
    import pdb; pdb.set_trace()
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    clustering_methods = [KMeans]
    reduction_before = [True, False]
    accuracy_results = {"red_before": {
                                "KMeans": None, 
                            }, 
                        "red_after": {
                                "KMeans": None, 
                            }
                        }
    for method in clustering_methods:
        for order in reduction_before:
            name = "before" if order else "after"
            results = dimensionality_reduction(X_train, y_train, X_test, y_test, name, method, red_before=order, verbose=False)
            if order:
                accuracy_results["red_before"][method.__name__] = results
            else:
                accuracy_results["red_after"][method.__name__] = results

    with open(f"./results/clustering_acc.json", 'w') as f:
        json.dump(accuracy_results, f, sort_keys=True, indent=4)
        f.write('\n')

# TODO: autoencoder and umap,