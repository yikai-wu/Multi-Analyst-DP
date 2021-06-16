import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from src.fair_experiments import error_calc, strategy_comp
from src.hdmm import workload, matrix
import src.census_workloads as census
from src.workload_selection import workload_selection

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

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
k_max = args.k
n = 64
eps = args.eps
rep = args.rep
seed = args.seed
t = args.t
if seed is not None:
    np.random.seed(seed)
print(experiment_name)
print('n={}, k_max={}, eps={}, rep={}, seed={}, t={}'.format(n,k_max,eps,rep,seed,t))
conf = OrderedDict()
conf['n']=n
conf['k_max']=k_max
conf['eps'] = eps
conf['rep']=rep
conf['seed'] =seed
conf['t'] = t
modes = ['ind', 'iden', 'uni', 'fmax', 'fsum', 'buc_con', 'buc_qeq', 'buc_qsd']
W_name = np.array(['race1', 'race2', 'white', 'Total', 'Identity', 'Prefix', 'H2'])
W_lst = np.array([census.__race1(), census.__race2(), census.__white(), workload.Total(n), workload.Identity(n), workload.Prefix(n), workload.H2(n)])
A_lst = strategy_comp(W_lst, n, rep)
results = []
names = []
ks = []
total_errors = pd.DataFrame()
mean_ratio_errors = pd.DataFrame()
max_ratio_errors = pd.DataFrame()
min_ratio_errors = pd.DataFrame()
max_distances = pd.DataFrame()
min_distances = pd.DataFrame()
gini_coefficients = pd.DataFrame()
mean_idenratio_errors = pd.DataFrame()
max_idenratio_errors = pd.DataFrame()
min_idenratio_errors = pd.DataFrame()
iden_gini_coefficients = pd.DataFrame()
for i in range(t):
    print(i, flush=True)
    k = np.random.randint(2, k_max+1)
    ks.append(k)
    Ws, Wn, As = workload_selection(W_lst, W_name, A_lst, n, k, rep)
    res = error_calc(Ws, Ws, n, eps, modes, rep, As=As, Ar=As)
    results.append(res)
    names.append(Wn)
    total_error = OrderedDict()
    mean_ratio_error = OrderedDict()
    max_ratio_error = OrderedDict()
    min_ratio_error = OrderedDict()
    max_distance = OrderedDict()
    min_distance = OrderedDict()
    gini_coefficient = OrderedDict()
    mean_idenratio_error = OrderedDict()
    max_idenratio_error = OrderedDict()
    min_idenratio_error = OrderedDict()
    iden_gini_coefficient = OrderedDict()
    for mode in modes:
        total_error[mode] = np.sum(res[mode])
        ratio_error = np.divide(res[mode], res['ind'])
        mean_ratio_error[mode] = np.mean(ratio_error)
        max_ratio_error[mode] = np.max(ratio_error)
        min_ratio_error[mode] = np.min(ratio_error)
        distance = np.subtract(res['ind'],res[mode])
        max_distance[mode] = np.max(distance)
        min_distance[mode] = np.min(distance)
        gini_coefficient[mode] = gini(ratio_error)
        idenratio_error = np.divide(res[mode], res['iden'])
        mean_idenratio_error[mode] = np.mean(idenratio_error)
        max_idenratio_error[mode] = np.max(idenratio_error)
        min_idenratio_error[mode] = np.min(idenratio_error)
        iden_gini_coefficient[mode] = gini(idenratio_error)

    total_errors = pd.concat([total_errors, pd.DataFrame(total_error, index=[i])])
    mean_ratio_errors = pd.concat([mean_ratio_errors, pd.DataFrame(mean_ratio_error, index=[i])])
    max_ratio_errors = pd.concat([max_ratio_errors, pd.DataFrame(max_ratio_error, index=[i])])
    min_ratio_errors = pd.concat([min_ratio_errors, pd.DataFrame(min_ratio_error, index=[i])])
    max_distances = pd.concat([max_distances, pd.DataFrame(max_distance, index=[i])])
    min_distances = pd.concat([min_distances, pd.DataFrame(min_distance, index=[i])])
    gini_coefficients = pd.concat([gini_coefficients, pd.DataFrame(gini_coefficient, index=[i])])
    mean_idenratio_errors = pd.concat([mean_idenratio_errors, pd.DataFrame(mean_idenratio_error, index=[i])])
    max_idenratio_errors = pd.concat([max_idenratio_errors, pd.DataFrame(max_idenratio_error, index=[i])])
    min_idenratio_errors = pd.concat([min_idenratio_errors, pd.DataFrame(min_idenratio_error, index=[i])])
    iden_gini_coefficients = pd.concat([iden_gini_coefficients, pd.DataFrame(iden_gini_coefficient, index=[i])])

names = np.asarray(names)
ks = np.asarray(ks)
out_dict=dict()
out_dict['conf'] = conf
out_dict['ks'] = ks
out_dict['names'] = names
out_dict['results'] = results
out_dict['total_errors'] = total_errors
out_dict['mean_ratio_errors'] = mean_ratio_errors
out_dict['max_ratio_errors'] = max_ratio_errors
out_dict['min_ratio_errors'] = min_ratio_errors
out_dict['max_distances'] = max_distances
out_dict['min_distances'] = min_distances
out_dict['gini_coefficients'] = gini_coefficients
out_dict['mean_idenratio_errors'] = mean_idenratio_errors
out_dict['max_idenratio_errors'] = max_idenratio_errors
out_dict['min_idenratio_errors'] = min_idenratio_errors
out_dict['iden_gini_coefficients'] = iden_gini_coefficients

file_name = experiment_name + '_k{}_t{}_2'.format(k_max, t)
file_path = os.path.join(data_path, file_name)
fw = open(file_path+'.pkl', 'wb')
pickle.dump(out_dict, fw)
fw.close()
print('Results saved')