import sys
import numpy as np
import pandas as pd
from math import sqrt
from numpy import random


# *** Argument reading and processing *** #

def read_arguments():
    K = 0
    flag_K = True

    try:
        K = int(sys.argv[1])
    except ValueError:
        flag_K = False

    iter = 300  # Default value.
    i = 4  # Represents the index of file_name_1 in the args array, assuming iter was provided.
    j = 5  # Represents the index of file_name_2 in the args array, assuming iter was provided.
    flag_iter = True

    if len(sys.argv) == 6:  # In case iter is provided
        try:
            iter = int(sys.argv[1])
        except ValueError:
            flag_iter = False

    else:  # Else, iter is not provided
        i -= 1
        j -= 1

    eps = float(sys.argv[i - 1])
    file_name_1 = sys.argv[i]
    file_name_2 = sys.argv[j]

    # Create DataFrames from the provided files
    if file_name_1[-4:] == ".txt":  # In case a txt file is provided
        data_1 = np.genfromtxt(file_name_1, dtype=float, encoding=None, delimiter=",")  # A two-dimensional numpy array
        df1 = pd.DataFrame(data_1)

    else:  # In case a csv file is provided
        df1 = pd.read_csv(file_name_1, header=None)

    if file_name_2[-4:] == ".txt":  # In case a txt file is provided
        data_2 = np.genfromtxt(file_name_2, dtype=float, encoding=None, delimiter=",")  # A two-dimensional numpy array
        df2 = pd.DataFrame(data_2)

    else:  # In case a csv file is provided
        df2 = pd.read_csv(file_name_2, header=None)

    return flag_K, flag_iter, K, iter, eps, df1, df2


def initialize():

    # Reading user arguments
    flag_K, flag_iter, K, iter, eps, df1, df2 = read_arguments()

    # Combine both input files by inner join using the first column in each file as a key
    data = pd.merge(df1, df2, on=df1.columns[0])

    data.columns = ['index'] + [f'x{i}' for i in range(1, data.shape[1])]

    data['index'] = data['index'].astype(int)
    data = data.set_index('index')

    data = data.sort_index()    # Sort the data points by the key in ascending order

    return flag_K, flag_iter, K, iter, eps, data


# *** Algorithm *** #

def check_arguments(flag_K, flag_iter, K, N, iter):

    flag_K = flag_K and 1 < K < N
    flag_iter = flag_iter and 1 < iter < 1000

    if not flag_K:
        print("Invalid number of clusters!")

    if not flag_iter:
        print("Invalid maximum iteration!")

    return flag_K and flag_iter


def dist(x1, x2):
    return sqrt(np.sum(np.square(x1 - x2)))


def D(x, centroids):
    return min(dist(x, y) for y in centroids)


def init_centroids(data, K):
    idx = data.index.tolist()
    N = data.shape[0]
    centroids = []
    centroids_index = []

    # Choose one center uniformly at random
    i = random.randint(N)
    new_centroid = data.iloc[i].values
    centroids.append(new_centroid)
    centroids_index.append(idx[i])

    # Repeat until K centers have been chosen
    for _ in range(1, K):
        distances = [D(row, centroids) for _, row in data.iterrows()]
        sum_dist = sum(distances)
        probabilities = [d / sum_dist for d in distances]

        l = np.random.choice(range(N), 1, p=probabilities)[0]
        new_centroid = data.iloc[l].values
        centroids.append(new_centroid)
        centroids_index.append(idx[l])

    return centroids, centroids_index


def print_centroids(centroids, centroids_index):
    str1 = ','.join(map(str, centroids_index))
    print(str1)

    for centroid in centroids:
        str1 = ','.join(map(lambda x: "%.4f" % x, centroid))
        print(str1)


def k_means_pp_algorithm():
    flag_K, flag_iter, K, iter, eps, data = initialize()
    valid_args = check_arguments(flag_K, flag_iter, K, data.shape[0], iter)

    if valid_args:
        np.random.seed(0)
        centroids, centroids_index = init_centroids(data, K)
        # centroids, centroids_index = mykmeanssp.fit(data, centroids, iter, eps)

        print_centroids(centroids, centroids_index)


k_means_pp_algorithm()
