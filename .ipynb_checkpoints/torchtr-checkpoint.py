import torch
import math
import numpy as np

def _pFactors(N):
    """Finds the prime factors of 'N'""" 
    from math import sqrt 
    pFact, limit, check, num = [], int(sqrt(N)) + 1, 2, N 
    if N == 1:
        return [1]
    for check in range(2, limit): 
         while num % check == 0: 
            pFact.append(check) 
            num /= check 
    if num > 1: 
      pFact.append(int(num)) 
    return pFact 

class TR():
    # té els cores de la TR decomp
    def __init__(self, source, shape, rank):
        # factors = _pFactors(N)
        print("Source:", source)
        tr_shape = _pFactors(np.prod(shape))
        print(tr_shape)

def randn(rank, shape):
    return TR(torch.rand(shape), shape, rank)