import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc, strategy_comp
from src.hdmm import workload, matrix
from src.workload_selection import workload_selection
from src.interference import interference_custom

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

results_path = './results/'
data_path = './data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=int, default=6)
parser.add_argument('-p', type=int, default=2)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-t', type=int, default=10)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int)
args = parser.parse_args()

run = args.run
k_max = args.k
d = args.d
p = args.p
n = p**d
eps = args.eps
rep = args.rep
seed = args.seed
t = args.t
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('d={}, p={}, n={}, k_max={}, eps={}, rep={}, seed={}, t={}'.format(d, p, n, k_max,eps,rep,seed,t))
conf = OrderedDict()
conf['d'] = d
conf['p'] = p
conf['n']=n
conf['k_max']=k_max
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed
conf['t'] = t
modes = ['ind', 'iden', 'uni', 'fsum', 'buc_con', 'buc_qsd']

W_name = []
W_lst = []
domain = np.full(d, p, dtype=np.int)
for i in range(d+1):
    key = 2**i
    W_name.append(key)
    W_lst.append(workload.Marginal(domain, key))
A_lst = strategy_comp(W_lst, n, rep)

results = []
names = []
ks = []
total_errors = pd.DataFrame()
max_ratio_errors = pd.DataFrame()
inters = pd.DataFrame()

for i in range(t):
    print(i, flush=True)
    k = np.random.randint(2, k_max+1)
    ks.append(k)
    Ws, Wn, As = workload_selection(W_lst, W_name, A_lst, n, k, rep, prob=1)
    error_base, total_error, max_ratio_error, inter = interference_custom(Ws, As, n, eps, modes, rep)
    results.append(error_base)
    names.append(Wn)
    total_errors = pd.concat([total_errors, pd.DataFrame(total_error, index=[i])])
    max_ratio_errors = pd.concat([max_ratio_errors, pd.DataFrame(max_ratio_error, index=[i])])
    inters = pd.concat([inters, pd.DataFrame(inter, index=[i])])

names = np.asarray(names)
ks = np.asarray(ks)
out_dict=dict()
out_dict['conf'] = conf
out_dict['ks'] = ks
out_dict['names'] = names
out_dict['results'] = results
out_dict['total_errors'] = total_errors
out_dict['max_ratio_errors'] = max_ratio_errors
out_dict['inters'] = inters

file_name = experiment_name + '_n{}_k{}_t{}_run{}'.format(n, k_max, t, run)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')