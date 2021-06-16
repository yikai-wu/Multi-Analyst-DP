import os, sys, argparse
import numpy as np
from src.hdmm import workload, matrix, error

A1 = matrix.EkteloMatrix(np.array([[2,0],[0,1]]))
A2 = matrix.EkteloMatrix(np.array([[1,0],[0,1]]))
A3 = matrix.EkteloMatrix(np.array([[1,0],[1,0],[0,1]]))
W1 = matrix.EkteloMatrix(np.array([[1,0]]))
W2 = matrix.EkteloMatrix(np.array([[1,0],[0,1]]))
W3 = matrix.EkteloMatrix(np.array([[0,1]]))

print('W1:')
print(error.expected_error(W1, W1, eps=1))
print(error.expected_error(W1, A1, eps=1))
print(error.expected_error(W1, A2, eps=1))
print(error.expected_error(W2, A3, eps=1))

print('W2:')
print(error.expected_error(W1,W1,eps=1)+error.expected_error(W3,W3,eps=1/2))
print(error.expected_error(W2, A1, eps=1))
print(error.expected_error(W2, A2, eps=1))
print(error.expected_error(W2, A3, eps=1))

print('A3:')
print(error.expected_error(A3, A3, eps=1))
print(error.expected_error(A3, A1, eps=1))
print(error.expected_error(A3, A2, eps=1))

print('W3:')
print(error.expected_error(W3, A2, eps=2))
print(error.expected_error(W3, A1, eps=3))