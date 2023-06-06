import sys
import pandas as pd
import numpy as np


### Argument reading and processing ###

def read_cmd_arguments():
    K = 0
    flag_K = True
    try:
        K = int(sys.argv[1])
    except ValueError:
        flag_K = False
    iter = 300  # default value.
    i = 4  # represents the index of file_name_1 in the args array, assuming iter was provided.
    j = 5  # represents the index of file_name_2 in the args array, assuming iter was provided.
    flag_iter = True
    if len(sys.argv) == 6:  # iter was provided
        try:
            iter = int(sys.argv[1])
        except ValueError:
            flag_iter = False
    else:  # iter was not provided
        i -= 1
        j -= 1

    eps = int(sys.argv[i - 1])
    data_points1 = pd.read_csv(sys.argv[i], header=None)
    data_points2 = pd.read_csv(sys.argv[j], header=None)

    return flag_K, flag_iter, K, iter, eps, data_points1, data_points2


def initialize():

    ### Reading user CMD arguments ###
    flag_K, flag_iter, K, iter, eps, data_points1, data_points2 = read_cmd_arguments()

    ### Combine both input files by inner join using the first column in each file as a key ###
    key = data_points1.columns.values.tolist()[0]
    data_points = pd.merge(data_points1, data_points2, on=key)

    ### After join, sort the data points by the 'key' in ascending order ###
    data_points = data_points.sort_values(by=key, ascending=True)

    return flag_K, flag_iter, K, iter, eps, data_points


### Algorithm ###

def check_arguments(flag_K, flag_iter, K, N, iter):
    flag_K = flag_K and 1 < K < N
    if flag_K:
        pass
    else:
        print("Invalid number of clusters!")

    flag_iter = flag_iter and 1 < iter < 1000
    if flag_iter:
        pass
    else:
        print("Invalid maximum iteration!")

    return flag_K and flag_iter


def smart_initialization_of_centroids(data_points):
    # convert datapoints to list of lists of float.
    data_points = data_points.values.tolist()
    # convert datapoints to numpy array.
    data_points = np.array(data_points)

    centroids = []

    return centroids


def k_means_pp_algorithm():
    flag_K, flag_iter, K, iter, eps, data_points = initialize()
    arguments_ok = check_arguments(flag_K, flag_iter, K, data_points.shape[0], iter)

    if arguments_ok:
        ### Implementation of the k-means++ algorithm ###
        np.random.seed(0)
        initial_centroids = smart_initialization_of_centroids(data_points)

        print(initial_centroids)


k_means_pp_algorithm()
