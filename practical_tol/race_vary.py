import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc, strategy_comp
from src.hdmm import workload, matrix
import src.census_workloads as census
from src.workload_selection import workload_selection
from src.interference import interference_tol

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

results_path = './results/'
data_path = './data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=20)
parser.add_argument('-ty', type=int, default=3)
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-t', type=int, default=10)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-tol', type=float, default=1e-2)
parser.add_argument('-seed', type=int)
args = parser.parse_args()

run = args.run
k_max = args.k
n = 64
eps = args.eps
rep = args.rep
seed = args.seed
t = args.t
ty = args.ty
tol = args.tol
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k_max={}, eps={}, rep={}, seed={}, t={}, ty={}, tol={}'.format(n,k_max,eps,rep,seed,t,ty, tol))
conf = OrderedDict()
conf['n']=n
conf['k_max']=k_max
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed
conf['t'] = t
conf['ty'] = ty
conf['tol'] = tol
modes = ['ind', 'iden', 'uni', 'fsum', 1, 0.999, 0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]
#modes = ['ind', 1.0]
W_name = np.array(['race1', 'race2', 'white', 'Total', 'Identity', 'Prefix', 'H2'])
W_lst = np.array([census.__race1(), census.__race2(), census.__white(), workload.Total(n), workload.Identity(n), workload.Prefix(n), workload.H2(n)])
A_lst = strategy_comp(W_lst, n, rep)
if ty == 3:
    types = ['random', 'range', 'single']
else:
    types = ['random', 'range', 'single', 'zero_one']

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
    Ws, Wn, As = workload_selection(W_lst, W_name, A_lst, n, k, rep, types=types, prob=7/8)
    error_base, total_error, max_ratio_error, inter = interference_tol(Ws, As, n, eps, modes, rep)
    results.append(error_base)
    names.append(Wn)
    total_errors = pd.concat([total_errors, pd.DataFrame(total_error, index=[i])])
    max_ratio_errors = pd.concat([max_ratio_errors, pd.DataFrame(max_ratio_error, index=[i])])
    inters = pd.concat([inters, pd.DataFrame(inter, index=[i])])

names = np.asarray(names)
ks = np.asarray(ks)
out_dict=dict()
out_dict['conf'] = conf
out_dict['types'] = types
out_dict['ks'] = ks
out_dict['names'] = names
out_dict['results'] = results
out_dict['total_errors'] = total_errors
out_dict['max_ratio_errors'] = max_ratio_errors
out_dict['inters'] = inters

file_name = experiment_name + '_k{}_t{}_ty{}_tol{}_run{}'.format(k_max, t, len(types), tol, run)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')