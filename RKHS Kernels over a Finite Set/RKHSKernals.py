import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt

##KERNELS
def k_polynomial(x, xp, d):
    return (np.dot(x, xp)+1)**d

def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x-xp)**2)/(2*(sigma**2)))

def k_tanh(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

def k_bessel(x, xp, n, v, sigma):
    return jv(v, sigma*np.sum((x-xp)**2))/(np.sum((x-xp)**2)**(-float(n*(v+1))))

def k_min(x, xp):
    return np.min([x, xp])

def kernel_mat(f, x):
    ''' Given a kernel f and a collection of observations x, construct the matrix
    [K]_{ij} = f(x_i, x_j)
    :param f: kernel function
    :param x: vector of values
    :return: K symmetric matrix
    '''
    n = len(x)
    K = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i+1):
            v = f(x[i], x[j])
            K[i, j] = v
            K[j, i] = v
    return K

def sample_functions(m, N, kernel):
    ''' For X={-m,...,1,0,1,...,m}, sample functions based on kernel
    :param m: +/- X range
    :param N: number of samples
    :param kernel: kernel function K(x,y)
    '''
    # Sample Functions
    X = np.arange(-m, m+1)
    K = kernel_mat(kernel, X)
    return X, np.random.multivariate_normal(np.zeros((2*m+1)), K, (N))

def plot_samples(ax, X, f, title):
    """ Given a plt.axis object, plot f and give it a title
    :param ax: pyplot axis object
    :param f: n samples of length l
    :param title: string for axis title
    """
    # Plot Results
    for fi in f:
        ax.plot(X, fi)
    ax.set_title(title)
    ax.set_ylabel(r"$f_i(x)$")
    ax.set_xlim((np.min(X), np.max(X)))

    def main():
        m = 10  # +/- range of X
        tau = 2  # tau Gaussian kernel parameter
        N = 10  # Number of samples

        X, f0 = sample_functions(m, N, lambda x, y: k_gaussian(x, y, 10))
        X, f1 = sample_functions(m, N, lambda x, y: k_gaussian(x, y, 30))
        X, f2 = sample_functions(m, N, lambda x, y: k_polynomial(x, y, 7))
        X, f3 = sample_functions(m, N, lambda x, y: k_min(y, x))

        # Plot Results
        figg, axg = plt.subplots((2), sharex=True, figsize=(4, 8))
        figk, axk = plt.subplots((2), sharex=True, figsize=(4, 8))
        plot_samples(axg[0], X, f0, r"Samples for $\tau = 10$")

        plot_samples(axg[1], X, f1, r"Samples for $\tau = 30$")
        axg[1].set_xlabel(r"$x$")

        plot_samples(axk[1], X, f3, r"Samples for $K(x, y) =\operatorname{min}(y, x)$")
        plot_samples(axk[0], X, f2, r"Samples for Polynomial Kernel $d=7$")
        axk[1].set_xlabel(r"$x$")

        plt.figure(figg.number)
        plt.savefig("gaussian_kernel_fig.eps")
        plt.figure(figk.number)
        plt.savefig("other_kernel_fig.eps")
        plt.show()

    if __name__ == "__main__":
        main()