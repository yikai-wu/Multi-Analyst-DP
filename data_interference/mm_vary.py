import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc, strategy_comp
from src.hdmm import workload, matrix
from src.interference import interference_data

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
parser.add_argument('-run', type=str, default=0)
parser.add_argument('-t', type=int, default=10)
parser.add_argument('-eps', type=float, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int)
parser.add_argument('-x', type=str, default='covid.csv')
parser.add_argument('-sam', type=int, default=100)
args = parser.parse_args()

run = args.run
k_max = args.k
eps = args.eps
rep = args.rep
seed = args.seed
sample = args.sam
t = args.t
x_file = args.x
x_data = pd.read_csv(x_file, header=None).to_numpy()
n = x_data.shape[0]

if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k_max={}, eps={}, rep={}, seed={}, t={}, sample={}, x_file={}'.format(n, k_max,eps,rep,seed,t,sample, x_file))
conf = OrderedDict()
conf['n']=n
conf['k_max']=k_max
conf['sample']=sample
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed
conf['t'] = t
conf['x_file'] = x_file
modes = ['ind', 'iden', 'uni', 'fsum', 'buc_qsd']

x_mean = np.sum(x_data[:,0]*x_data[:,1])/np.sum(x_data[:,1])
mid = np.sum(x_data[:,1])/2
prefix = np.cumsum(x_data[:,1])
x_median = x_data[np.searchsorted(prefix, mid),0]
x_mode = x_data[np.argmax(x_data[:,1]),0]

meanW = np.vstack((x_data[:,0], np.ones(n)))
meanW = matrix.EkteloMatrix(meanW)
W_name = np.array(['Mean', 'Median'])
W_lst = np.array([meanW, workload.Prefix(n)])
A_lst = strategy_comp(W_lst, n, rep)
q_lst = W_name

ans_lst = np.array([x_mean, x_median, x_mode])
print(ans_lst)

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
    ls = np.random.choice(len(W_lst), size=k)
    Wn = W_name[ls]
    names.append(Wn)
    error_base, total_error, max_ratio_error, inter = interference_data(W_lst, A_lst, ls, n, eps, modes, rep, qk=q_lst, x=x_data, ansk=ans_lst, sample=sample)
    results.append(error_base)
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

file_name = experiment_name + '_k{}_t{}_eps{}_sam{}_run{}'.format(k_max, t, eps, sample, run)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')