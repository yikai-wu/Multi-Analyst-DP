import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc
from src.hdmm import workload, matrix
from src.interference import interference

data_path = './data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-n', type=int, default=16)
parser.add_argument('-k', type=int, default=10)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int)
args = parser.parse_args()

run_number = args.run
n = args.n
k = args.k
eps = args.eps
rep = args.rep
seed = args.seed
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k={}, eps={}, rep={}, seed={}'.format(n,k,eps,rep,seed))
conf = OrderedDict()
conf['n']=n
conf['k']=k
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed

outs =[]
modes = ['ind', 'uni', 'fdiff', 'fmax', 'fsum', 'buc_eq', 'buc_con', 'buc_qeq', 'buc_qsd']
W1 = workload.Identity(n)
W2 = workload.Total(n)
W_lst = [W1, W2]
ls = np.hstack((0,np.ones(k-1, dtype=int)))
print(ls)


err_diff, err_inter, err_diff_ratio, err_inter_ratio, err_diff_max, err_diff_ratio_max, err_inter_max, err_inter_ratio_max = interference(W_lst, ls, n, eps, modes, rep)
print(err_diff, err_inter, err_diff_ratio, err_inter_ratio, err_inter_max, err_inter_ratio_max)
out_dict=dict()
out_dict['conf'] = conf
out_dict['differences'] = pd.DataFrame(err_diff, index=[0])
out_dict['interferences'] = pd.DataFrame(err_inter, index=[0])
out_dict['diff_ratios'] = pd.DataFrame(err_diff_ratio, index=[0])
out_dict['inter_ratios'] = pd.DataFrame(err_inter_ratio, index=[0])
out_dict['diff_maxs'] = pd.DataFrame(err_diff_max, index=[0])
out_dict['diff_ratio_maxs'] = pd.DataFrame(err_diff_ratio_max, index=[0])
out_dict['inter_maxs'] = pd.DataFrame(err_inter_max, index=[0])
out_dict['inter_ratio_maxs'] = pd.DataFrame(err_inter_ratio_max, index=[0])

file_name = experiment_name+'_n{}_k{}'.format(n,k)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')