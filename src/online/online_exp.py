import os, sys, argparse
import random
import src.hdmm.workload as workload
import src.census_workloads as census
import src.online.online_workloads as online_workloads
import numpy as np
from collections import OrderedDict

results_path = '/Users/albertsun/onlineHDMM/results/'
data_path = '/Users/albertsun/onlineHDMM/data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

# run the file with the following script: 
# /Users/albertsun/opt/anaconda3/envs/py310/bin/python -m src.online.online_exp -h
# replace the first thing with your own python environment

def crossmode_analysis(res1, res2):
    out = OrderedDict()
    distance = res1 - res2
    out['total_utility'] = np.sum(distance)
    out['max_utility'] = np.max(distance)
    out['min_utility'] = np.min(distance)
    if out['min_utility'] < 1e-6:
        print('!!Violates sharing incentive!!')
        print(res1, res2)
        out['sharing_incentive'] = 'False'
    else:
        out['sharing_incentive'] = 'True'
    return out

def interference_analysis(res1, res2):
    out = OrderedDict()
    distance = res1 - res2
    out['total_utility'] = np.sum(distance)
    out['max_utility'] = np.max(distance)
    out['min_utility'] = np.min(distance)
    if out['min_utility'] < 1e-6:
        print('!!Violates non-interference!!')
        print(res1, res2)
        out['non_interference'] = 'False'
    else:
        out['non_interference'] = 'True'
    return out

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=10)
parser.add_argument('-run', type=str, default=1)
parser.add_argument('-eps', type=int, default=1)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-seed', type=int)
args = parser.parse_args()

run = args.run
k = args.k
n = 64
eps = args.eps
rep = args.rep
seed = args.seed
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k={}, eps={}, rep={}, seed={}'.format(n,k,eps,rep,seed))
conf = OrderedDict()
conf['n']=n
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed

# TODO: add prefix sum and custom
W_name = ['identity', 'total', 'H2', 'race1', 'race2', 'race3'] 
W_lst = [online_workloads.identity(n), online_workloads.total(n), online_workloads.H2(n), online_workloads.race1(), online_workloads.race2(), online_workloads.race3()]

c = np.random.choice(len(W_lst))
Ws = [W_lst[c]]
Wn = [W_name[c]]
#res = error_calc()



import random
k_max=20
k = random.randint(2, k_max)