import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
import math
import os
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import eigh
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home

#Bridge
seed = 0
N = 100
M = 10

numpy.random.seed(seed)

def sample_path_batch(M, N):
    dt = 1.0 / N
    dt_sqrt = numpy.sqrt(dt)
    B = numpy.empty((M, N), dtype=numpy.float32)
    B[:, 0] = 0
    for n in six.moves.range(N - 1):
        t = n * dt
        xi = numpy.random.randn(M) * dt_sqrt
        B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
    return B

B = sample_path_batch(M, N)
pyplot.plot(B.T)
pyplot.show()