import os, sys, argparse
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

from .fair_experiments import error_calc, strategy_comp, error_calc_tol
from .hdmm import workload, matrix

def interference(Wk, Ak, ls, n, eps, modes, rep):
    uls, uid, ucnt = np.unique(ls, return_index=True, return_counts=True)
    k = len(ls)
    Ws = np.take(Wk, ls)
    Wr = np.take(Wk, uls)
    As = np.take(Ak, ls)
    Ar = np.take(Ak, uls)
    err_base = error_calc(Ws, Wr, n, eps, modes, rep, As=As, Ar=Ar)
    err_diff_tots = OrderedDict()
    err_inter_tots = OrderedDict()
    err_diff_ratios = OrderedDict()
    err_inter_ratios = OrderedDict()
    err_diff_maxs = OrderedDict()
    err_diff_ratio_maxs = OrderedDict()
    err_inter_maxs = OrderedDict()
    err_inter_ratio_maxs = OrderedDict()
    for mode in modes:
        err_diff_tots[mode] = 0
        err_inter_tots[mode] = 0
        err_diff_ratios[mode] = 0
        err_inter_ratios[mode] = 0
        err_diff_maxs[mode] = 0
        err_diff_ratio_maxs[mode] = 0
        err_inter_maxs[mode] = 0
        err_inter_ratio_maxs[mode] = 0
    for i in range(len(uls)):
        Ws_new = np.delete(Ws, uid[i])
        As_new = np.delete(As, uid[i])
        err_new = error_calc(Ws_new, Wr, n, eps*(k-1)/k, modes, rep, As=As_new, Ar=Ar)
        for mode in modes:
            err_diff = err_base[mode] - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_diff_tots[mode] += (np.sum(err_diff*ucnt)-err_diff[i])*ucnt[i]
            err_diff_ratios[mode] += (np.sum(err_diff_ratio*ucnt)-err_diff_ratio[i])*ucnt[i]
            err_inter = np.maximum(err_diff, 0)
            err_inter_ratio = np.maximum(err_diff_ratio, 0)
            err_inter_tots[mode] += (np.sum(err_inter*ucnt)-err_inter[i])*ucnt[i]
            err_inter_ratios[mode] += (np.sum(err_inter_ratio*ucnt)-err_inter_ratio[i])*ucnt[i]
            err_diff_maxs[mode] = np.maximum(err_diff_maxs[mode], np.amax(err_diff))
            err_diff_ratio_maxs[mode] = np.maximum(err_diff_ratio_maxs[mode], np.amax(err_diff_ratio))
            err_inter_maxs[mode] = np.maximum(err_inter_maxs[mode], np.amax(err_inter))
            err_inter_ratio_maxs[mode] = np.maximum(err_inter_ratio_maxs[mode], np.amax(err_inter_ratio))

    for mode in modes:
        err_diff_tots[mode] /= (k-1)*k
        err_inter_tots[mode] /= (k-1)*k
        err_diff_ratios[mode] /= (k-1)*k
        err_inter_ratios[mode] /= (k-1)*k
    return err_diff_tots, err_inter_tots, err_diff_ratios, err_inter_ratios, err_diff_maxs, err_diff_ratio_maxs, err_inter_maxs, err_inter_ratio_maxs

def interference_nonfix(Wk, ls, n, eps, modes, rep):
    uls, uid, ucnt = np.unique(ls, return_index=True, return_counts=True)
    k = len(ls)
    Ws = np.take(Wk, ls)
    Wr = np.take(Wk, uls)
    err_base = error_calc(Ws, Wr, n, eps, modes, rep)
    err_diff_tots = OrderedDict()
    err_inter_tots = OrderedDict()
    err_diff_ratios = OrderedDict()
    err_inter_ratios = OrderedDict()
    err_inter_maxs = OrderedDict()
    err_inter_ratio_maxs = OrderedDict()
    for mode in modes:
        err_diff_tots[mode] = 0
        err_inter_tots[mode] = 0
        err_diff_ratios[mode] = 0
        err_inter_ratios[mode] = 0
        err_inter_maxs[mode] = 0
        err_inter_ratio_maxs[mode] = 0
    for i in range(len(uls)):
        Ws_new = np.take(Wk, np.delete(ls, uid[i]))
        err_new = error_calc(Ws_new, Wr, n, eps*(k-1)/k, modes, rep)
        for mode in modes:
            err_diff = err_base[mode] - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_diff_tots[mode] += (np.sum(err_diff*ucnt)-err_diff[i])*ucnt[i]
            err_diff_ratios[mode] += (np.sum(err_diff_ratio*ucnt)-err_diff_ratio[i])*ucnt[i]
            err_inter = np.maximum(err_diff, 0)
            err_inter_ratio = np.maximum(err_diff_ratio, 0)
            err_inter_tots[mode] += (np.sum(err_inter*ucnt)-err_inter[i])*ucnt[i]
            err_inter_ratios[mode] += (np.sum(err_inter_ratio*ucnt)-err_inter_ratio[i])*ucnt[i]
            err_inter_maxs[mode] = np.maximum(err_inter_maxs[mode], np.amax(err_inter))
            err_inter_ratio_maxs[mode] = np.maximum(err_inter_ratio_maxs[mode], np.amax(err_inter_ratio))

    for mode in modes:
        err_diff_tots[mode] /= (k-1)*k
        err_inter_tots[mode] /= (k-1)*k
        err_diff_ratios[mode] /= (k-1)*k
        err_inter_ratios[mode] /= (k-1)*k
    return err_diff_tots, err_inter_tots, err_diff_ratios, err_inter_ratios, err_inter_maxs, err_inter_ratio_maxs

def interference_new(Wk, Ak, ls, n, eps, modes, rep):
    uls, uid, ucnt = np.unique(ls, return_index=True, return_counts=True)
    k = len(ls)
    Ws = np.take(Wk, ls)
    Wr = np.take(Wk, uls)
    As = np.take(Ak, ls)
    Ar = np.take(Ak, uls)
    err_base = error_calc(Ws, Wr, n, eps, modes, rep, As=As, Ar=Ar)
    err_inters = OrderedDict()
    total_error = OrderedDict()
    max_ratio_error = OrderedDict()

    for mode in modes:
        err_inters[mode] = 0
        total_error[mode] = np.sum(err_base[mode])
        ratio_error = np.divide(err_base[mode], err_base['ind'])
        max_ratio_error[mode] = np.max(ratio_error)

    for i in range(len(uls)):
        Ws_new = np.delete(Ws, uid[i])
        As_new = np.delete(As, uid[i])
        err_new = error_calc(Ws_new, Wr, n, eps*(k-1)/k, modes, rep, As=As_new, Ar=Ar)
        for mode in modes:
            err_diff = err_base[mode] - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_inters[mode] = np.maximum(err_inters[mode], np.amax(err_diff_ratio))
    return err_base, total_error, max_ratio_error, err_inters

def interference_custom(Ws, As, n, eps, modes, rep, tol=1e-2):
    err_base = error_calc(Ws, Ws, n, eps, modes, rep, As=As, Ar=As, tol=tol)
    err_inters = OrderedDict()
    total_error = OrderedDict()
    max_ratio_error = OrderedDict()
    k = len(Ws)
    for mode in modes:
        err_inters[mode] = 0
        total_error[mode] = np.sum(err_base[mode])
        ratio_error = np.divide(err_base[mode], err_base['ind'])
        max_ratio_error[mode] = np.max(ratio_error)
        
    for i in range(len(Ws)):
        Ws_new = np.delete(Ws, i)
        As_new = np.delete(As, i)
        err_new = error_calc(Ws_new, Ws_new, n, eps*(k-1)/k, modes, rep, As=As_new, Ar=As_new, tol=tol)
        for mode in modes:
            err_diff = np.delete(err_base[mode], i) - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_inters[mode] = np.maximum(err_inters[mode], np.amax(err_diff_ratio))
    return err_base, total_error, max_ratio_error, err_inters

def interference_data(Wk, Ak, ls, n, eps, modes, rep, qk=None, x=None, ansk=None, sample=100, tol=1e-2):
    uls, uid = np.unique(ls, return_index=True)
    k = len(ls)
    Ws = np.take(Wk, ls)
    Wr = np.take(Wk, uls)
    As = np.take(Ak, ls)
    Ar = np.take(Ak, uls)
    qr = np.take(qk, uls)
    ansr = np.take(ansk, uls)
    err_base = error_calc(Ws, Wr, n, eps, modes, rep, As=As, Ar=Ar, qr=qr, x=x, ans=ansr, sample=sample, tol=tol)
    err_inters = OrderedDict()
    total_error = OrderedDict()
    max_ratio_error = OrderedDict()

    for mode in modes:
        err_inters[mode] = 0
        total_error[mode] = np.sum(err_base[mode])
        ratio_error = np.divide(err_base[mode], err_base['ind'])
        max_ratio_error[mode] = np.max(ratio_error)

    for i in range(len(uls)):
        Ws_new = np.delete(Ws, uid[i])
        As_new = np.delete(As, uid[i])
        err_new = error_calc(Ws_new, Wr, n, eps*(k-1)/k, modes, rep, As=As_new, Ar=Ar, qr=qr, x=x, ans=ansr, sample=sample, tol=tol)
        for mode in modes:
            err_diff = err_base[mode] - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_inters[mode] = np.maximum(err_inters[mode], np.amax(err_diff_ratio))
    return err_base, total_error, max_ratio_error, err_inters

def interference_tol(Ws, As, n, eps, modes, rep):
    err_base = error_calc_tol(Ws, Ws, n, eps, modes, rep, As=As, Ar=As)
    err_inters = OrderedDict()
    total_error = OrderedDict()
    max_ratio_error = OrderedDict()
    k = len(Ws)
    for mode in modes:
        err_inters[mode] = 0
        total_error[mode] = np.sum(err_base[mode])
        ratio_error = np.divide(err_base[mode], err_base['ind'])
        max_ratio_error[mode] = np.max(ratio_error)
        
    for i in range(len(Ws)):
        Ws_new = np.delete(Ws, i)
        As_new = np.delete(As, i)
        err_new = error_calc_tol(Ws_new, Ws_new, n, eps*(k-1)/k, modes, rep, As=As_new, Ar=As_new)
        for mode in modes:
            err_diff = np.delete(err_base[mode], i) - err_new[mode]
            err_diff_ratio = np.divide(err_diff, err_new[mode])
            err_inters[mode] = np.maximum(err_inters[mode], np.amax(err_diff_ratio))
    return err_base, total_error, max_ratio_error, err_inters