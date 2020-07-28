import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.linalg import eigh
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture

def gauss_kernel_old(X,gamma):

    X = np.reshape(X,(len(X),1))

    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-1.0*(mat_sq_dists) / ( 2*gamma**2 ) )

    return K

def gauss_kernel(x1,x2,tau=1):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * (tau ** 2)))

def polynomial_kernel(x1,x2,p=2):
    return np.dot(x1,x2)**p

def compute_kernel(X,kernel):
    N = X.shape[0]
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i][j] = kernel(X[i],X[j])

    return K


def kernel_em_cluster(X, kernel, k):
    '''
    :param X:Data set
    :param K:Kernel of choice
    :param k:number of clusters
    :param iters: max iterations
    :return: labels, cluster means
    '''

    K = compute_kernel(X, kernel)

    eigvalues, eigvectors = eigh(K)

    S = np.column_stack((eigvectors[:, -i] for i in range(1, k + 1)))

    gmm = GaussianMixture(n_components=k).fit(S)

    labels = gmm.predict(S)

    plt.scatter(S[:,0],S[:,1])
    plt.show()

    cluster_1 = np.where(labels == 0)
    cluster_2 = np.where(labels == 1)
    # plot first cluster
    plt.scatter(S[cluster_1, 0], S[cluster_1, 1], color='blue')
    plt.scatter(S[cluster_2, 0], S[cluster_2, 1], color='red')

    plt.show()

    return labels


def kernel_k_means(X,kernel,k,iters=100):
    '''
    :param X:Data set
    :param K:Kernel of choice
    :param k:number of clusters
    :param iters: max iterations
    :return: labels, cluster means
    '''

    K = compute_kernel(X,kernel)

    eigvalues , eigvectors = eigh(K)

    S = np.column_stack((eigvectors[:,-i] for i in range(1,k+1)))

    kmeans = KMeans(n_clusters=k,random_state=25).fit(S)

    cluster_1 = np.where(kmeans.labels_ == 0)
    cluster_2 = np.where(kmeans.labels_ == 1)
    # plot first cluster
    plt.scatter(S[cluster_1,0],S[cluster_1,1],color='blue')
    plt.scatter(S[cluster_2,0],S[cluster_2,1],color='red')

    plt.scatter(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1],marker='x',color='blue')
    plt.scatter(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1],marker='x',color='red')

    plt.show()

    return kmeans.labels_ , kmeans.cluster_centers_


if __name__ == '__main__':

    print('Running kernel k-means')

    data_path = '/Users/befeltingu/Downloads/kkm1.csv'

    data = pd.read_csv(data_path).as_matrix()[:,1:]

    plt.scatter(data[:,0],data[:,1])

    plt.show()

    labels, centers = kernel_k_means(data,gauss_kernel,2)

    #labels = kernel_em_cluster(data,polynomial_kernel,2)

    cluster_1 = np.where(labels == 0)
    cluster_2 = np.where(labels == 1)
    # plot first cluster
    plt.scatter(data[cluster_1, 0], data[cluster_1, 1], color='blue')
    plt.scatter(data[cluster_2, 0], data[cluster_2, 1], color='red')

    plt.show()

    print('Done running kernel k-means')


