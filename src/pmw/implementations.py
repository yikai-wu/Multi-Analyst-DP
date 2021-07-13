"""
Different implementations of private multiplicative weights.
Author: Albert Sun, 7/13/2021
"""

import numpy as np
import math
from matplotlib import pyplot as plt


def pmw(W, x, eps=0.01, beta=0.1):
    """
    Implement Private Multiplicative Weights Mechanism (PMW) on a workload of
    linear queries with constants equal to exactly how the Hardt and Rothblum
    2010 theoretical paper implement PMW.

    - W = workload of queries (M x k numpy array)
    - x = true database (M x 1 numpy array)
    """

    print(f'original database: {x}')
    print(f'workload: \n{W}, size = {np.shape(W)}')

    M = x.size  # len of database, num of coordinates in the DB
    n = x.sum()  # sum of database
    k = len(W)  # num of queries
    delta = 1 / (n * math.log(n, np.e))

    x_norm = x / np.sum(x)

    eta = math.log(M, np.e) ** (1 / 4) / math.sqrt(n)
    sigma = 10 * math.log(1 / delta, np.e) * (math.log(M, np.e)) ** (1 / 4) / (
            math.sqrt(n) * eps)
    T = 4 * sigma * (math.log(k, np.e) + math.log(1 / beta, np.e))  # threshold

    # initialize synthetic database at time 0 (prior to any queries)
    y_t = np.ones(M) / M
    x_t = np.ones(M) / M  # fractional histogram computed in round t

    # append to list of databases y_t and x_t
    y_list = [y_t]
    x_list = [x_t]

    update_count = 0
    query_answers = []

    # iterate through time = (0, k)
    for t, query in enumerate(W):

        # compute noisy answer by adding Laplacian noise
        A_t = np.random.laplace(loc=0, scale=sigma, size=1)[0]
        a_t_hat = np.dot(query, x) + A_t
        # print(f'a_t_hat: {a_t_hat}')

        # compute difference between noisy answer and answer from maintained
        # histogram
        d_t_hat = a_t_hat - np.dot(query, x_list[t])

        # lazy round: use already maintained histogram to answer the query
        if abs(d_t_hat) <= T:
            query_answers.append(np.dot(query, x_list[t]))
            x_list.append(x_list[t])
            continue

        # update round: update the histogram and return the noisy answer,
        # abs(d_t_hat) > T
        else:
            update_count += 1
            # step a
            r_t = np.zeros(M)
            if d_t_hat > 0:
                r_t = query
            else:
                r_t = np.ones(M) - query
            for i in range(len(x_t)):
                y_t[i] = x_list[t][i] * math.exp(-eta * r_t[i])
            y_list.append(y_t)

            # step b
            x_t = y_t / np.sum(y_t)
            x_list.append(x_t)

        if update_count > n * math.log(M, 10) ** (1 / 2):
            return "failure"
        else:
            query_answers.append(a_t_hat / np.sum(x))

    # calculate absolute error (L1)
    real_ans = np.matmul(W, x_norm)
    error = np.abs(query_answers - real_ans)
    print(f'error: {error}')

    # plot absolute error
    x_axis = range(1, k + 1)
    plt.title('Error:')
    plt.xticks(x_axis)
    plt.plot(x_axis, error, label='Absolute Error')

    # calculate relative error
    relative_error = np.abs(query_answers / real_ans)
    print(f'relative error: {relative_error}')

    # plot relative error
    plt.xticks(x_axis)
    plt.plot(x_axis, relative_error, label='Relative Error')

    print(f'T (Threshold) = {T}')
    print(f'query_answers (using pmw): {query_answers}\n')

    print(
        f'The update threshold for failure is n * math.log(M, 10)**(1/2): '
        f'{n * math.log(M, 10) ** (1 / 2)}. n is {n}, and M is {M}.')
    print(f'Update Count = {update_count}\n')

    print(f'Updated Database = {x_t}')

    return query_answers


def pmw_optimized(W, x, eps=0.01, beta=0.1, laplace_scale=1, threshold=100):
    """
    Implement Private Multiplicative Weights Mechanism (PMW) on a workload of
    linear queries. New arguments to allow for optimizing the amount of
    privacy budget used in each step.

    - W = workload of queries (M x k numpy array)
    - x = true database (M x 1 numpy array)
    """

    print(f'original database: {x}')
    print(f'workload: \n{W}, size = {np.shape(W)}')

    M = x.size  # len of database, num of coordinates in the DB
    n = x.sum()  # sum of database
    k = len(W)  # num of queries
    delta = 1 / (n * math.log(n, np.e))

    x_norm = x / np.sum(x)

    eta = math.log(M, np.e) ** (1 / 4) / math.sqrt(n)
    sigma = 10 * math.log(1 / delta, np.e) * (math.log(M, np.e)) ** (1 / 4) / (
            math.sqrt(n) * eps)
    T = 4 * sigma * (math.log(k, np.e) + math.log(1 / beta, np.e))  # threshold

    # initialize synthetic database at time 0 (prior to any queries)
    y_t = np.ones(M) / M
    x_t = np.ones(M) / M  # fractional histogram computed in round t

    # append to list of databases y_t and x_t
    y_list = [y_t]
    x_list = [x_t]

    update_count = 0
    query_answers = []

    # iterate through time = (0, k)
    for t, query in enumerate(W):

        # compute noisy answer by adding Laplacian noise
        A_t = np.random.laplace(loc=0, scale=sigma, size=1)[0]
        a_t_hat = np.dot(query, x) + A_t
        # print(f'a_t_hat: {a_t_hat}')

        # compute difference between noisy answer and answer from maintained
        # histogram
        d_t_hat = a_t_hat - np.dot(query, x_list[t])

        # lazy round: use already maintained histogram to answer the query
        if abs(d_t_hat) <= T:
            query_answers.append(np.dot(query, x_list[t]))
            x_list.append(x_list[t])
            continue

        # update round: update the histogram and return the noisy answer,
        # abs(d_t_hat) > T
        else:
            update_count += 1
            # step a
            r_t = np.zeros(M)
            if d_t_hat > 0:
                r_t = query
            else:
                r_t = np.ones(M) - query
            for i in range(len(x_t)):
                y_t[i] = x_list[t][i] * math.exp(-eta * r_t[i])
            y_list.append(y_t)

            # step b
            x_t = y_t / np.sum(y_t)
            x_list.append(x_t)

        if update_count > n * math.log(M, 10) ** (1 / 2):
            return "failure"
        else:
            query_answers.append(a_t_hat / np.sum(x))

    # calculate absolute error (L1)
    real_ans = np.matmul(W, x_norm)
    error = np.abs(query_answers - real_ans)
    print(f'error: {error}')

    # plot absolute error
    x_axis = range(1, k + 1)
    plt.title('Absolute Error (L1):')
    plt.xticks(x_axis)
    plt.plot(x_axis, error)

    # calculate relative error
    relative_error = np.abs(query_answers / real_ans)
    print(f'relative error: {relative_error}')

    print(f'T (Threshold) = {T}')
    print(f'query_answers (using pmw): {query_answers}\n')

    print(
        f'The update threshold for failure is n * math.log(M, 10)**(1/2): '
        f'{n * math.log(M, 10) ** (1 / 2)}. n is {n}, and M is {M}.')
    print(f'Update Count = {update_count}\n')

    print(f'Updated Database = {x_t}')

    return query_answers