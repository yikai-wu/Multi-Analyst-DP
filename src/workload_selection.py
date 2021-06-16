import os, sys, argparse
import numpy as np
from collections import OrderedDict

from .fair_experiments import error_calc, strategy_comp
from .hdmm import workload, matrix

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

def random_workload_generation(n, min_size=1, max_size=None, types=None):
    if types == None:
        types = ['random', 'range', 'single', 'zero_one']
    switcher = {'random': random_query,
                'range': range_query,
                'single': single_query,
                'zero_one': zero_one_query}
    if max_size is None:
        max_size = 2*n
    
    m = np.random.randint(min_size, max_size+1)
    W = []
    for i in range(m):
        c = np.random.randint(len(types))
        q = switcher[types[c]](n)
        W.append(q)
    W = np.stack(W)
    return matrix.EkteloMatrix(W)

def workload_selection(W_lst, W_name, A_lst, n, k, rep, prob=0.5, types=None):
    Ws = []
    As = []
    Wn = []
    for i in range(k):
        if np.random.random() < prob:
            c = np.random.randint(len(W_lst))
            Ws.append(W_lst[c])
            Wn.append(W_name[c])
            As.append(A_lst[c])
        else:
            W = random_workload_generation(n, types=types)
            A = strategy_comp([W], n, rep)[0]
            Ws.append(W)
            Wn.append('custom')
            As.append(A)
    return Ws, Wn, As