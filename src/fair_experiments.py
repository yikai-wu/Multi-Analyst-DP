import numpy as np
from .hdmm import workload, fairtemplates, error, fairmechanism, matrix, mechanism, templates
from . import bucket_filling as buc
import scipy
import numbers
import pandas as pd

def mean_ans(y, x_ind):
    return y[0]/y[1]

def median_ans(y, x_ind):
    mid = y[-1]/2.0
    loc = np.searchsorted(y, mid)
    if loc >= len(x_ind):
        loc = len(x_ind)-1
    return x_ind[loc]

def mode_ans(y,x_ind):
    return x_ind[np.argmax(y)]

def percentile_ans(y, x_ind, per):
    loc = y[-1]*(per/100.0)
    loc = np.searchsorted(y, loc)
    if loc >= len(x_ind):
        loc = len(x_ind)-1
    return x_ind[loc]

def expected_error_data(W, A, eps=1.0, query='Linear', x=None, ans=None, sample=100):
    switcher = {
        'Median': median_ans,
        'Mode': mode_ans,
        'Mean': mean_ans
    }
    if query in switcher.keys():
        x_ind = x[:,0]
        x_data = matrix.EkteloMatrix(np.expand_dims(x[:,1],axis=1))
        Wx = W @ x_data
        m = A.shape[0]
        delta = A.sensitivity()
        WAi = W @ A.pinv()
        WAi = (delta/eps) * WAi
        err_sum = 0
        for i in range(sample):
            b = np.random.laplace(size=(m,1))
            b = matrix.EkteloMatrix(b)
            y = Wx + WAi @ b
            y = y.dense_matrix()
            y = np.array(y).flatten()
            ans_h = switcher.get(query)(y, x_ind)
            err = (ans-ans_h)**2
            err_sum += err
        return np.maximum(err_sum/sample,1e-6)
    elif isinstance(query, str) and 'Per_' in query:
        x_ind = x[:,0]
        x_data = matrix.EkteloMatrix(np.expand_dims(x[:,1],axis=1))
        per = float(query.split('_')[-1])
        Wx = W @ x_data
        m = A.shape[0]
        delta = A.sensitivity()
        WAi = W @ A.pinv()
        WAi = (delta/eps) * WAi
        err_sum = 0
        for i in range(sample):
            b = np.random.laplace(size=(m,1))
            b = matrix.EkteloMatrix(b)
            y = Wx + WAi @ b
            y = y.dense_matrix()
            y = np.array(y).flatten()
            ans_h = percentile_ans(y, x_ind, per)
            err = (ans-ans_h)**2
            err_sum += err
        return np.maximum(err_sum/sample,1e-6)
    else:
        return error.expected_error(W, A, eps)

def strategy_comp_sub(Ws, n):
    losses = []  
    As = []
    for W in Ws:
        pid = templates.PIdentity(max(1, n//16), n)
        fun, loss = pid.optimize(W)
        As.append(pid.strategy())
        losses.append(loss)
    return np.asarray(As), np.asarray(losses)

def strategy_comp(Ws, n, rep):
    As_best, losses_min = strategy_comp_sub(Ws, n)
    for i in range(1,rep):
        As, losses = strategy_comp_sub(Ws, n)
        idx = np.less(losses, losses_min)
        losses_min[idx] = losses[idx]
        As_best[idx] = As[idx]
    return As_best

def independent_error(Ws, Wr, n, eps, rep, Ar=None, qr=None, x=None, ans=None, sample=100):
    if Ar is None:
        Ar = strategy_comp(Wr, n, rep)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, Ar[i], eps=eps/len(Ws), query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def unified_error(Ws, Wr, n, eps, rep, As=None, qr=None, x=None, ans=None, sample=100):
    Wt = workload.VStack(Ws)
    pid_best = templates.PIdentity(max(1, n//16), n)
    _, loss_min = pid_best.optimize(Wt)
    for i in range(1, rep):
        pid = templates.PIdentity(max(1, n//16), n)
        _, loss = pid.optimize(Wt)
        if loss < loss_min:
            pid_best = pid
            loss_min = loss
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, pid_best.strategy(), eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def fairmax_error(Ws, Wr, n, eps, rep, As=None, qr=None, x=None, ans=None, sample=100):
    pid_best = fairtemplates.PIdentity(max(1, n//16), n, mode='max')
    _, loss_min = pid_best.optimize(Ws, indAs=As)
    for i in range(1,rep):
        pid = fairtemplates.PIdentity(max(1, n//16), n, mode='max')
        _, loss = pid.optimize(Ws, indAs=As)
        if loss < loss_min:
            pid_best = pid
            loss_min = loss
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, pid_best.strategy(), eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def fairsum_error(Ws, Wr, n, eps, rep, As=None, qr=None, x=None, ans=None, sample=100):
    pid_best = fairtemplates.PIdentity(max(1, n//16), n, mode='sum')
    _, loss_min = pid_best.optimize(Ws, indAs=As)
    for i in range(1,rep):
        pid = fairtemplates.PIdentity(max(1, n//16), n, mode='sum')
        _, loss = pid.optimize(Ws, indAs=As)
        if loss < loss_min:
            pid_best = pid
            loss_min = loss
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, pid_best.strategy(), eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def fairdiff_error(Ws, Wr, n, eps, rep, As=None, qr=None, x=None, ans=None, sample=100):
    pid_best = fairtemplates.PIdentity(max(1, n//16), n, mode='diff')
    _, loss_min = pid_best.optimize(Ws, indAs=As)
    for i in range(1,rep):
        pid = fairtemplates.PIdentity(max(1, n//16), n, mode='diff')
        _, loss = pid.optimize(Ws, indAs=As)
        if loss < loss_min:
            pid_best = pid
            loss_min = loss
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, pid_best.strategy(), eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def identity_error(Ws, Wr, n, eps, rep, As=None, qr=None, x=None, ans=None, sample=100):
    A = matrix.Identity(n)
    errors = []
    for i, W in enumerate(Wr):
        errors.append(expected_error_data(W, A, eps=eps, query=qr[i], x=x, ans=ans[i], sample=sample))
    return np.asarray(errors)

def error_calc(Ws, Wr, n, eps, modes, rep, As=None, Ar=None, qr=None, x=None, ans=None, sample=100, tol=1e-2):
    switcher = {
        'uni': unified_error,
        'fdiff': fairdiff_error,
        'fmax': fairmax_error,
        'fsum': fairsum_error,
        'iden': identity_error
    }
    if qr is None:
        qr = np.zeros(len(Wr))
    if ans is None:
        ans = np.zeros(len(Wr))
    result = dict()
    buc_modes = []
    for mode in modes:
        if mode == 'ind':
            errors = independent_error(Ws, Wr, n, eps, rep, Ar=Ar, qr=qr, x=x, ans=ans, sample=sample)
            result[mode] = errors
        elif 'buc_' in mode:
            buc_modes.append(mode)
        else:
            errors = switcher.get(mode)(Ws, Wr, n, eps, rep, As=As, qr=qr, x=x, ans=ans, sample=sample)
            result[mode] = errors
    if buc_modes:     
        buc_result = buc.buc_error_calc(Ws, Wr, n, eps, buc_modes, rep, As=As, qr=qr, x=x, ans=ans, sample=sample, tol=tol)
    result.update(buc_result)
    return result

def error_calc_tol(Ws, Wr, n, eps, modes, rep, As=None, Ar=None, qr=None, x=None, ans=None, sample=100):
    switcher = {
        'uni': unified_error,
        'fdiff': fairdiff_error,
        'fmax': fairmax_error,
        'fsum': fairsum_error,
        'iden': identity_error
    }
    if qr is None:
        qr = np.zeros(len(Wr))
    if ans is None:
        ans = np.zeros(len(Wr))
    result = dict()
    buc_modes = []
    for mode in modes:
        if mode == 'ind':
            errors = independent_error(Ws, Wr, n, eps, rep, Ar=Ar, qr=qr, x=x, ans=ans, sample=sample)
            result[mode] = errors
        elif isinstance(mode, numbers.Number):
            buc_modes.append(mode)
        else:
            errors = switcher.get(mode)(Ws, Wr, n, eps, rep, As=As, qr=qr, x=x, ans=ans, sample=sample)
            result[mode] = errors
    if buc_modes:     
        buc_result = buc.water_tol_error_calc(Ws, Wr, n, eps, buc_modes, rep, As=As, qr=qr, x=x, ans=ans, sample=sample)
    result.update(buc_result)
    return result