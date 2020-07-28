import numpy as np
import matplotlib.pyplot as plt
import os

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
from scipy.linalg import eigh

def center_kernel(K):
    '''

    :param K: Kernel
    :return: Center Kernel
    '''
    N = K.shape[0]

    one_n = np.ones((N, N)) / N

    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    return K

def linear_kernel(x,y):
    return np.dot(x,y)

def polynomial_kernel(x,y,p=4):
    return (linear_kernel(x,y))**p

def compute_kernel(X,kernel):
    N = X.shape[0]
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i][j] = kernel(X[i],X[j])

    return K

def kpca(X,kernel,n_components):
    '''

    :param X: The data
    :param gamma:
    :param n_components: How many eignvectors to return
    :return:
    '''

    K = compute_kernel(X,kernel)

    K = center_kernel(K)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


if __name__ == '__main__':

    run_mnist_example = 1
    if run_mnist_example:

        fetch_mnist()
        from sklearn.datasets import fetch_mldata

        mnist = fetch_mldata("MNIST original")

        ALPHA = 0.1

        X = mnist['data'].copy()

        X = X.reshape((len(X), -1))

        X = X / 255.0

        y = mnist['target'].copy()

        digit_indexes = np.where((y == 1) | (y == 7))

        sub_indexes = np.random.choice(digit_indexes[0], 500, replace=False)

        sub_x = X[sub_indexes]

        sub_y = y[sub_indexes]

        n_samples, n_features = X.shape

        mnist_kpca = kpca(sub_x,polynomial_kernel, 8)

        for i in range(0, 4):
            digit_1_1 = mnist_kpca[[sub_y == 1, 2 * i]]
            digit_1_2 = mnist_kpca[[sub_y == 1, 2 * i + 1]]
            digit_7_1 = mnist_kpca[[sub_y == 7, 2 * i]]
            digit_7_2 = mnist_kpca[[sub_y == 7, 2 * i + 1]]

            plt.scatter(digit_1_1, digit_1_2, color='red')
            plt.scatter(digit_7_1, digit_7_2, edgecolors='blue')
            plt.xlabel('PC{}'.format(str(2 * i + 1)))
            plt.ylabel('PC{}'.format(str(2 * i + 2)))
            plt.show()

        print('Done running the KPCA MNIST example')

