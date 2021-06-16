import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import strategy_comp
from src.hdmm import workload, matrix
from src.interference import interference_new

data_path = './data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-n', type=int, default=16)
parser.add_argument('-k', type=int, default=10)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int, default=0)
args = parser.parse_args()

run_number = args.run
n = args.n
k = args.k
eps = args.eps
rep = args.rep
seed = args.seed
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
modes = ['ind', 'uni', 'fmax', 'fsum', 'buc_con', 'buc_qsd']
W1 = workload.Identity(n)
W2 = workload.Total(n)
W_lst = [W1, W2]
A_lst = strategy_comp(W_lst, n, rep)
ls = np.hstack((0,np.ones(k-1, dtype=int)))
print(ls)



err_base, total_error, max_ratio_error, err_inter = interference_new(W_lst, A_lst, ls, n, eps, modes, rep)
out_dict=dict()
out_dict['conf'] = conf
out_dict['error'] = err_base
out_dict['total_errors'] = total_error
out_dict['max_ratio_errors'] = max_ratio_error
out_dict['inters'] = err_inter

file_name = experiment_name+'_n{}_k{}_s{}'.format(n,k,seed)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')