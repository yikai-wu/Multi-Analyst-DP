import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc
from src.hdmm import workload, matrix
import src.census_workloads as census

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

results_path = '/usr/xtmp/yw267/FairHDMM/census/results/'
data_path = '/usr/xtmp/yw267/FairHDMM/census/data/'
experiment_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=10)
parser.add_argument('-run', type=str, default=1)
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
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k={}, eps={}, rep={}, seed={}'.format(n,k,eps,rep,seed))
conf = OrderedDict()
conf['n']=n
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed

modes = ['ind', 'util', 'WUitl', 'AEgal', 'MEgal']
W_name = ['adult', 'age1', 'age2', 'age3', 'Total', 'Identity']
W_lst = [census.__adult(), census.__age1(), census.__age2(), census.__age3(), workload.Total(n), workload.Identity(n)]
c = np.random.choice(len(W_lst))
Ws = [W_lst[c]]
Wn = [W_name[c]]
res = error_calc(Ws, Ws, n, 1/k*eps, modes, rep)
analysis = pd.DataFrame()
result = pd.DataFrame.from_dict(res, orient='index')
result = result.set_index(result.index+'_1')
results = result
out_dict = dict()
out_dict['res_1'] = res
res_prev = res

for j in range(2,k+1):
    outs = []
    index =[]
    c = np.random.choice(len(W_lst))
    Ws.append(W_lst[c])
    Wn.append(W_name[c])
    res = error_calc(Ws, Ws, n, j/k*eps, modes, rep)
    for mode in modes[1:]:
        print(mode)
        outs.append(crossmode_analysis(res['ind'], res[mode]))
        outs.append(interference_analysis(res_prev[mode], res[mode][:-1]))
        index.extend([mode+'_{}_ind'.format(j), mode+'_{}_inter'.format(j)])
    analysis = pd.concat([analysis, pd.DataFrame(outs, index=index)])
    result = pd.DataFrame.from_dict(res, orient='index')
    result = result.set_index(result.index+'_{}'.format(j))
    results = pd.concat([results, result])
    out_dict['res_{}'.format(j)] = res
    res_prev = res
names = pd.DataFrame([Wn], index=['names'])
results = pd.concat([results, names])
print(results)

out_dict['conf'] = conf
out_dict['Ws'] = Ws
out_dict['Wn'] = Wn
out_dict['results'] = results
out_dict['analysis'] = analysis

file_name = experiment_name + '_k{}_run{}'.format(k, run)
file_path = os.path.join(results_path, file_name)
results.to_csv(file_path+'_res.csv')
analysis.to_csv(file_path+'_an.csv')
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')