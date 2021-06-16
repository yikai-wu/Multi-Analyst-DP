import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc
from src.hdmm import workload, matrix

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

results_path = './results/'
data_path = '/usr/xtmp/yw267/FairHDMM/adversarial/data/'
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
W1 = workload.AllRange(n)
W2 = workload.Identity(n)
Ws = [W1]
for i in range(1,k):
    Ws.append(W2)
Wr = Ws[:2]

outs = []
index =[]
res = error_calc(Ws, Wr, n, eps, modes, rep)
res_noW1 = error_calc(Ws[1:], Wr[1:], n, eps*(k-1)/k, modes, rep)
for mode in modes[1:]:
    print(mode)
    outs.append(crossmode_analysis(res['ind'], res[mode]))
    outs.append(interference_analysis(res_noW1[mode], res[mode][1:]))
    index.extend([mode+'_ind', mode+'_inter'])
analysis = pd.DataFrame(outs, index=index)
results = pd.DataFrame.from_dict(res, orient='index')
results_noW1 = pd.DataFrame.from_dict(res_noW1, orient='index')
results_noW1 = results_noW1.set_index(results_noW1.index+'_noW1')
results_noW1.insert(len(results_noW1.columns), len(results_noW1.columns), np.zeros(len(results_noW1)))
results_noW1 = results_noW1.shift(1,axis=1)
results = results.append(results_noW1)
print(results)

out_dict = dict()
out_dict['conf'] = conf
out_dict['Ws'] = Ws
out_dict['Wr'] = Wr
out_dict['res'] = res
out_dict['res_noW1'] = res_noW1
out_dict['results'] = results
out_dict['analysis'] = analysis

file_name = experiment_name+'_n{}_k{}'.format(n,k)
file_path = os.path.join(results_path, file_name)
results.to_csv(file_path+'_res.csv')
analysis.to_csv(file_path+'_an.csv')
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')