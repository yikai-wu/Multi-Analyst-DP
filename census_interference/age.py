import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import strategy_comp
from src.hdmm import workload, matrix
import src.census_workloads as census
from src.interference import interference

results_path = './results/'
data_path = './data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=10)
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-t', type=int, default=10)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int)
args = parser.parse_args()

run = args.run
k = args.k
n = 115
eps = args.eps
rep = args.rep
seed = args.seed
t = args.t
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k={}, eps={}, rep={}, seed={}, t={}'.format(n,k,eps,rep,seed,t))
conf = OrderedDict()
conf['n']=n
conf['k']=k
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed
conf['t'] = t
modes = ['ind', 'uni', 'fdiff', 'fmax', 'fsum', 'buc_con', 'buc_qeq', 'buc_qsd']
W_name = np.array(['adult', 'age1', 'age2', 'age3', 'Total', 'Identity'])
W_lst = np.array([census.__adult(), census.__age1(), census.__age2(), census.__age3(), workload.Total(n), workload.Identity(n)])
A_lst = strategy_comp(W_lst, n, rep)
names = []
err_differences = pd.DataFrame()
err_interferences = pd.DataFrame()
err_diff_ratios = pd.DataFrame()
err_inter_ratios = pd.DataFrame()
err_diff_maxs = pd.DataFrame()
err_diff_ratio_maxs = pd.DataFrame()
err_inter_maxs = pd.DataFrame()
err_inter_ratio_maxs = pd.DataFrame()
for i in range(t):
    print(i, flush=True)
    c = np.random.choice(len(W_lst), size=k)
    Wn = W_name[c]
    names.append(Wn)
    err_diff, err_inter, err_diff_ratio, err_inter_ratio, err_diff_max, err_diff_ratio_max, err_inter_max, err_inter_ratio_max = interference(W_lst, A_lst, c, n, eps, modes, rep)
    err_differences = pd.concat([err_differences, pd.DataFrame(err_diff, index=[i])])
    err_interferences = pd.concat([err_interferences, pd.DataFrame(err_inter, index=[i])])
    err_diff_ratios = pd.concat([err_diff_ratios, pd.DataFrame(err_diff_ratio, index=[i])])
    err_inter_ratios = pd.concat([err_inter_ratios, pd.DataFrame(err_inter_ratio, index=[i])])
    err_diff_maxs = pd.concat([err_diff_maxs, pd.DataFrame(err_diff_max, index=[i])])
    err_diff_ratio_maxs = pd.concat([err_diff_ratio_maxs, pd.DataFrame(err_diff_ratio_max, index=[i])])
    err_inter_maxs = pd.concat([err_inter_maxs, pd.DataFrame(err_inter_max, index=[i])])
    err_inter_ratio_maxs = pd.concat([err_inter_ratio_maxs, pd.DataFrame(err_inter_ratio_max, index=[i])])

names = np.asarray(names)
out_dict=dict()
out_dict['conf'] = conf
out_dict['names'] = names
out_dict['differences'] = err_differences
out_dict['interferences'] = err_interferences
out_dict['diff_ratios'] = err_diff_ratios
out_dict['inter_ratios'] = err_inter_ratios
out_dict['diff_maxs'] = err_inter_maxs
out_dict['diff_ratio_maxs'] = err_inter_ratio_maxs
out_dict['inter_maxs'] = err_inter_maxs
out_dict['inter_ratio_maxs'] = err_inter_ratio_maxs

file_name = experiment_name + '_k{}_t{}'.format(k, t)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')