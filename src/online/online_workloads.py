import src.hdmm.workload as workload
import src.census_workloads as census
from src.workload_selection import workload_selection
import numpy as np

def identity(n):
    return workload.Identity(n).dense_matrix()

def total(n):
    return workload.Total(n).dense_matrix()

def H2(n):
    return workload.H2(n).dense_matrix()

def race1():
    return census.__race1().dense_matrix()
    
def race2():
    return census.__race2().dense_matrix()
    
def race3():
    return census.__white().dense_matrix()

def random_query(n, max_weight=4):
    w = np.random.randint(1, max_weight+1)
    return w * np.random.rand(n)

def range_query(n):
    c = np.random.choice(n, 2, replace=False)
    l = np.min(c)
    r = np.max(c)
    q = np.zeros(n)
    q[np.arange(l,r+1)] = 1.0
    return q

def zero_one_query(n):
    w = np.random.randint(1, n)
    c = np.random.choice(n, w, replace=False)
    q = np.zeros(n)
    q[c] = 1.0
    return q

def single_query(n):
    c = np.random.randint(n)
    q = np.zeros(n)
    q[c] = 1.0
    return q

def custom(n):
    """
    Workload of length uniformly random from
    [1, 2n]. Each query in the workload is uniformly
    sampled from the set of {range, singleton, sum, 
    random queries}
    """
    switcher = {'random': random_query,
                'range': range_query,
                'single': single_query,
                'zero_one': zero_one_query}

    types = ['random', 'range', 'single', 'zero_one']

    m = np.random.randint(1, 2 * n+1)
    W = []
    for i in range(m):
        c = np.random.randint(len(types))
        q = switcher[types[c]](n)
        W.append(q)
    W = np.stack(W)
    return W

def prefix_sum(n):
    return workload.Prefix(n).dense_matrix()

