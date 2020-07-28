import math
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.linalg import eigh
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home

def fetch_mnist(data_home=None):
    '''
    Function to get original mnist data set if data does not already
    exist locally
    :param data_home:
    :return:
    '''
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


def sample_mnist(target_image=7.,sample_size=500,top_n=16,fig_per_row=3):

    mnist = fetch_mldata("MNIST original")

    mnist_data = mnist.data

    mnist_targets = mnist.target

    digit_indexes = list(np.where(mnist_targets == target_image)[0])

    sample_indexes = np.random.choice(digit_indexes, sample_size)

    sample = mnist_data[sample_indexes]

    mu = sample.mean(axis=0)

    K = np.cov(sample,rowvar=False)

    eigvalues , eigvectors = eigh(K)

    plt.imshow(np.reshape(mu,(28,28)))

    plt.title('Mean of the sample')
    plt.show()

    top_u = np.column_stack((eigvectors[:,-i] for i in range(1,top_n+1)))

    # create sub plot with matplotlib and
    num_rows = math.ceil(TOP_N / float(fig_per_row))
    fig = plt.figure()
    for i in range(1,TOP_N + 1):

        fig.add_subplot(num_rows,fig_per_row,i)

        img = np.reshape(top_u[:,i - 1],(28,28))

        plt.imshow(img)
        plt.axis('off')
        plt.title('u{}'.format(i))

    fig.tight_layout()
    plt.show()

    print('Done running mnist mvn sample')



if __name__ == '__main__':

    ################
    # GLOBAL PARAMS
    ################

    MNIST_IMAGE = 7. # which image to view
    SAMPLE_SIZE = 500 # how many images to sample
    TOP_N = 16 # how many eigen vectors to display
    FIG_PER_ROW = 4 # number of images to display per row

    # Get data if not already present locally
    fetch_mnist()
    from sklearn.datasets import fetch_mldata

    sample_mnist(MNIST_IMAGE,SAMPLE_SIZE,top_n=TOP_N,fig_per_row=FIG_PER_ROW)

    print('Done running MVN mnist sampling')
