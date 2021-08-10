"""
Different implementations of private multiplicative weights.

Author: Albert Sun, 7/13/2021
"""

import numpy as np
import math
from matplotlib import pyplot as plt


def plot_error(abs_error, rel_error, k, update_times):
    """Plot absolute and relative error"""
    plt.xticks(range(0, k, 5))
    plt.title('Error across queries:')
    rel_line, = plt.plot(rel_error, label='Relative Error')
    abs_line, = plt.plot(abs_error, label='Absolute Error')
    for xc in update_times:
        plt.axvline(x=xc, color='red', label='Update Times', linestyle='dashed')
    plt.legend(handles=[rel_line, abs_line])


def pmw(workload, x, eps=0.01, beta=0.1):
    """Implement Private Multiplicative Weights Mechanism (PMW) on a workload of
    linear queries with constants equal to the same constants in the Hardt and
    Rothblum 2010 theoretical paper implementation of PMW.

    - W = workload of queries (M x k numpy array)
    - x = true database (M x 1 numpy array)"""

    def print_outputs():
        # input
        print(f'original database: {x}')

        # updates and threshold
        print(f'T (Threshold) = {threshold}')
        print('Index \t Workload \t d_t_hat:')
        [print(index, query_and_error[0],
               query_and_error[1]) for index, query_and_error
         in enumerate(zip(workload, d_t_hat_list))]
        print(f'Update Count = {update_count}\n')

        # outputs
        print(f'query_answers (using pmw): {query_answers}\n')
        print(f'Updated Database = {x_t}')

    # initialize constants
    m = x.size  # database len
    n = x.sum()  # database sum
    k = len(workload)  # num of queries
    delta = 1 / (n * math.log(n, np.e))
    x_norm = x / np.sum(x)
    eta = math.log(m, np.e) ** (1 / 4) / math.sqrt(n)
    sigma = 10 * math.log(1 / delta, np.e) * (math.log(m, np.e)) ** (1 / 4) / (
            math.sqrt(n) * eps)
    threshold = 4 * sigma * (math.log(k, np.e) + math.log(1 / beta, np.e))

    # synthetic databases at time 0 (prior to any queries)
    y_t = np.ones(m) / m
    x_t = np.ones(m) / m

    # append to list of databases y_t and x_t
    y_list = [y_t]
    x_list = [x_t]

    update_count = 0
    query_answers = []
    update_times = []
    d_t_hat_list = []

    # iterate through time = (0, k)
    for time, query in enumerate(workload):

        # compute noisy answer by adding Laplacian noise
        a_t = np.random.laplace(loc=0, scale=sigma, size=1)[0]
        a_t_hat = np.dot(query, x) + a_t

        # difference between noisy and maintained histogram answer
        d_t_hat = a_t_hat - np.dot(query, x_list[time])
        d_t_hat_list.append(d_t_hat)

        # lazy round: use maintained histogram to answer the query
        if abs(d_t_hat) <= threshold:
            query_answers.append(np.dot(query, x_list[time]))
            x_list.append(x_list[time])
            continue

        # update round: update histogram and return noisy answer
        else:
            update_count += 1
            update_times.append(time)

            # step a
            if d_t_hat > 0:
                r_t = query
            else:
                r_t = np.ones(m) - query
            for i, v in enumerate(y_t):
                y_t[i] = x_list[time][i] * math.exp(-eta * r_t[i])
            y_list.append(y_t)

            # step b
            x_t = y_t / np.sum(y_t)
            x_list.append(x_t)

        if update_count > n * math.log(m, 10) ** (1 / 2):
            return "failure"
        else:
            query_answers.append(a_t_hat / np.sum(x))

    # calculate error
    real_ans = np.matmul(workload, x_norm)
    abs_error = np.abs(query_answers - real_ans)
    rel_error = np.abs(query_answers / np.where(real_ans == 0, 0.000001,
                                                real_ans))

    print_outputs()
    plot_error(abs_error, rel_error, k, update_times)

    return query_answers


def pmw_optimized(workload, x, eps=0.01, beta=0.1, laplace_scale=1,
                  threshold=100):
    """
    Implement Private Multiplicative Weights Mechanism (PMW) on a workload of
    linear queries. New arguments to allow for optimizing the amount of
    privacy budget used in each step.

    - W = workload of queries (M x k numpy array)
    - x = true database (M x 1 numpy array)
    """
    def print_outputs():
        # input
        print(f'original database: {x}')

        # updates and threshold
        print(f'T (Threshold) = {threshold}')
        print('Index \t Workload \t d_t_hat:')
        [print(index, query_and_error[0],
               query_and_error[1]) for index, query_and_error
         in enumerate(zip(workload, d_t_hat_list))]
        print(f'Update Count = {update_count}\n')

        # outputs
        print(f'query_answers (using pmw): {query_answers}\n')
        print(f'Updated Database = {x_t}')

    # initialize constants
    m = x.size  # database len
    n = x.sum()  # database sum
    k = len(workload)  # num of queries
    x_norm = x / np.sum(x)
    eta = math.log(m, np.e) ** (1 / 4) / math.sqrt(n)

    # synthetic databases at time 0 (prior to any queries)
    y_t = np.ones(m) / m
    x_t = np.ones(m) / m

    # append to list of databases y_t and x_t
    y_list = [y_t]
    x_list = [x_t]

    update_count = 0
    query_answers = []
    update_times = []
    d_t_hat_list = []
    # iterate through time = (0, k)
    for time, query in enumerate(workload):

        # compute noisy answer by adding Laplacian noise
        a_t = np.random.laplace(loc=0, scale=laplace_scale, size=1)[0]
        a_t_hat = np.dot(query, x) + a_t

        # difference between noisy and maintained histogram answer
        d_t_hat = a_t_hat - np.dot(query, x_list[time])
        d_t_hat_list.append(d_t_hat)

        # lazy round: use maintained histogram to answer the query
        if abs(d_t_hat) <= threshold:
            query_answers.append(np.dot(query, x_list[time]))
            x_list.append(x_list[time])
            continue

        # update round: update histogram and return noisy answer
        else:
            update_count += 1
            update_times.append(time)

            # step a
            if d_t_hat > 0:
                r_t = query
            else:
                r_t = np.ones(m) - query
            for i, v in enumerate(y_t):
                y_t[i] = x_list[time][i] * math.exp(-eta * r_t[i])
            y_list.append(y_t)

            # step b
            x_t = y_t / np.sum(y_t)
            x_list.append(x_t)

        if update_count > n * math.log(m, 10) ** (1 / 2):
            return "failure"
        else:
            query_answers.append(a_t_hat / np.sum(x))

    # calculate error
    real_ans = np.matmul(workload, x_norm)
    abs_error = np.abs(query_answers - real_ans)
    rel_error = np.abs(query_answers / np.where(real_ans == 0, 0.000001,
                                                real_ans))

    print_outputs()
    plot_error(abs_error, rel_error, k, update_times)

    return query_answers
