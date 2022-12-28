import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from matplotlib import cm
from sklearn.model_selection import train_test_split


def data_exploration(data):
    print(data.info())
    print(data.head())
    print("Presence of missing values: ", data.isnull().values.any())

def data_preprocessing(data_pd):
    data_np = data_pd.to_numpy()
    X = data_np[:, :-1]
    y = data_np[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test

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
    method_instance = method(n_components=components)
    x_dim_red = method_instance.fit_transform(x)
    return x_dim_red

def add_points_to_scatter_plot(ax, x, indices, color, title):
    """
    Adds the points specified in indices to the scatter plot in ax

    :param ax: scatter plot
    :param x: (ndarray(N, 256)) data
    :param indices: (ndarray(n,)) indices to select the data to be plotted
    :param color: (ndarray(4,)) RGBA color for the points
    """ 
    x_i = x[indices, :]
    ax.scatter(x_i[:,0], x_i[:,1], color=color)
    ax.set_xlabel(title.lower() + "_1", fontsize=30)
    ax.set_ylabel(title.lower() + "_2", fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    ax.set_title(title, fontsize=35, pad=10)
    return

def dimensionality_reduction(x, y, name, verbose = False):
    """
    Applies three different dimensionality reduction methods to the data
    """
    import pdb; pdb.set_trace()
    x_pca = dimensionality_reduction_fit_transform(PCA, 2, x)
    x_umap = dimensionality_reduction_fit_transform(umap.UMAP, 2, x)
    x_tsne = dimensionality_reduction_fit_transform(TSNE, 2, x)

    tab10 = cm.get_cmap('tab10')
    labels = np.unique(y)
    colors = tab10(labels)
    fig, (ax_pca, ax_umap, ax_tsne) = plt.subplots(1,3,figsize=(30, 11))
    for i, color in zip(labels, colors):
        indices = np.where(y == i)[0]
        add_points_to_scatter_plot(ax_pca, x_pca, indices, color, "PCA")
        add_points_to_scatter_plot(ax_umap, x_umap, indices, color, "UMAP")
        add_points_to_scatter_plot(ax_tsne, x_tsne, indices, color, "TSNE")
    fig.legend(labels, prop={'size': 30})
    title = "Dimensionality reduction " + name
    fig.suptitle(title, fontsize = 45, y=1)
    plt.savefig("results/" + title.replace(" ", "_")+ ".pdf")
    if verbose:
        plt.show()
        plt.close()
    return x_pca, x_umap, x_tsne

if __name__ == '__main__':
    data = pd.read_csv("A2_data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data)
    dimensionality_reduction(X_train, y_train, "hoal", verbose=True)
    import pdb; pdb.set_trace()
