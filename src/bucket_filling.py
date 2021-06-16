import numpy as np
from .hdmm import workload, fairtemplates, error, fairmechanism, matrix, mechanism, templates
from . import fair_experiments as fe
import scipy

def equality(Ws, Wr, As, n, eps, rep, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    analysts = [0]
    buckets = [0]
    budgets = [1]
    for i in range(1, len(As)):
        flag = True
        for j, bucket in enumerate(buckets):
            if np.allclose(As[i].dense_matrix(), As[bucket].dense_matrix(), rtol=tol, atol=1e-3):
                budgets[j] += 1
                analysts.append(j)
                flag = False
                break
        if flag:
            buckets.append(i)
            budgets.append(1)
            analysts.append(len(buckets)-1)

    errors = []
    for i, W in enumerate(Wr):
        idx = analysts[i]
        errors.append(fe.expected_error_data(W, As[buckets[idx]],eps=eps*budgets[idx]/len(Ws), query=qr[i], x=x, ans=ans[i], sample=sample))
    
    return np.asarray(errors)

def concat(Ws, Wr, As, n, eps, rep, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    A = matrix.VStack(As)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(fe.expected_error_data(W, A, eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)
    

def query_eq(Ws, Wr, As, n, eps, rep, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    buckets = []
    budgets = []
    for A in As:
        A = np.asarray(A.dense_matrix())
        for q in A:
            flag = True
            for i, b in enumerate(buckets):
                if np.allclose(q, b, rtol=tol, atol=1e-3):
                    budgets[i] += 1
                    flag = False
                    break
            if flag:
                buckets.append(q)
                budgets.append(1)
    A = np.asarray(buckets)
    budgets = np.expand_dims(np.asarray(budgets),axis=1)
    A = A*budgets/len(As)
    A = matrix.EkteloMatrix(A)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(fe.expected_error_data(W, A, eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    
    return np.asarray(errors)

def query_sd(Ws, Wr, As, n, eps, rep, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    buckets = []
    budgets = []
    for A in As:
        A = np.asarray(A.dense_matrix())
        for q in A:
            flag = True
            q_norm = np.linalg.norm(q)
            if q_norm == 0.0:
                continue
            q = q/q_norm
            for i, b in enumerate(buckets):
                if np.dot(q,b) >= 1-tol:
                    buckets[i] = b*budgets[i]+q*q_norm
                    budgets[i] = np.linalg.norm(buckets[i])
                    buckets[i] /= budgets[i]
                    flag = False
                    break
            if flag:
                buckets.append(q)
                budgets.append(q_norm)
    A = np.asarray(buckets)
    budgets = np.expand_dims(np.asarray(budgets),axis=1)
    A = A*budgets/len(As)
    A = matrix.EkteloMatrix(A)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(fe.expected_error_data(W, A, eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    
    return np.asarray(errors)

def query_sd2(Ws, Wr, As, n, eps, rep, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    buckets = []
    budgets = []
    for A in As:
        A = np.asarray(A.dense_matrix())
        for q in A:
            flag = True
            q_norm = np.linalg.norm(q)
            if q_norm == 0.0:
                continue
            q = q/q_norm
            for i, b in enumerate(buckets):
                if np.dot(q,b) >= 1-0.1*tol:
                    buckets[i] = b*budgets[i]+q*q_norm
                    budgets[i] = np.linalg.norm(buckets[i])
                    buckets[i] /= budgets[i]
                    flag = False
                    break
            if flag:
                buckets.append(q)
                budgets.append(q_norm)
    A = np.asarray(buckets)
    budgets = np.expand_dims(np.asarray(budgets),axis=1)
    A = A*budgets/len(As)
    A = matrix.EkteloMatrix(A)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(fe.expected_error_data(W, A, eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    
    return np.asarray(errors)

def buc_error_calc(Ws, Wr, n, eps, modes, rep, As=None, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    switcher = {
        'buc_eq': equality,
        'buc_con': concat,
        'buc_qeq': query_eq,
        'buc_qsd': query_sd,
        'buc_qsd2': query_sd2
    }
    if As is None:
        As = fe.strategy_comp(Ws, n, rep)
    result = dict()
    for mode in modes:
        error = switcher.get(mode)(Ws, Wr, As, n, eps, rep, qr=qr, x=x, ans=ans, sample=sample, tol=tol)
        result[mode] = error
    return result

def water_tol_error_calc(Ws, Wr, n, eps, tols, rep, As=None, qr=None, x=None, ans=None, sample=100):
    if As is None:
        As = fe.strategy_comp(Ws, n, rep)  
    result = dict()
    for tol in tols:
        error = query_sd(Ws, Wr, As, n, eps, rep, qr=qr, x=x, ans=ans, sample=sample, tol=tol)
        result[tol] = error
    return result