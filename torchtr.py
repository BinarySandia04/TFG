import torch
import math
import numpy as np
import itertools

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

def _unfolding_mat(tensor, mode):
    n, m = 1, 1
    for i in range(len(tensor.shape)):
        if i < mode:
            n *= tensor.shape[i]
        else:
            m *= tensor.shape[i]
    return torch.reshape(tensor, (n,m))

def _trsvd(tensor, rel_err):
    cores = []
    d = len(tensor.shape)
    unfolding_mat = _unfolding_mat(tensor, 1)
    # Truncated SVD a unfolding_mat
    U, S, Vh = torch.linalg.svd(unfolding_mat, full_matrices=False)
    
    p = 9000
    U_k = U[:, :p]  # Take first k columns
    S_k = torch.diag(S[:p])  # Take first k singular values
    Vh_k = Vh[:p, :]  # Take first k rows

    # Obtain ranks
    r = len(S_k)
    mr_1 = 1
    mn = 1 + r ** 2
    for i in range(1, r):
        if r % i == 0:
            if (r - i) ** 2 + i ** 2 < mn:
                mn = (r - i) ** 2 + i ** 2
                mr_1 = i
    mr_2 = r // mr_1
    
    cores.append(torch.permute(torch.reshape(U_k, [tensor.shape[0], mr_1, mr_2]), (1,0,2)))
    subchain = torch.permute(torch.reshape(S_k @ Vh_k, [mr_1, mr_2, -1]), (1, 2, 0))
    nr = mr_2
    oldr = mr_2
    
    for k in range(1, d):
        Z = torch.reshape(subchain, [oldr * tensor.shape[k], -1])

        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        U_k = U[:, :p]  # Take first k columns
        S_k = torch.diag(S[:p])  # Take first k singular values
        Vh_k = Vh[:p, :]  # Take first k rows
        
        nr = len(S_k)
        cores.append(torch.reshape(U_k, [oldr, tensor.shape[k], nr]))
        subchain = torch.reshape(S_k @ Vh_k, [nr, -1, mr_1])
        # subchain = torch.reshape(S_k, (1, -1)) * Vh_k
        oldr = nr
    # U, S, V = torch.linalg.svd(unfolding_mat)
    return cores

def _tr_get(tr, pos):
    p1 = tr[0].shape[0]
    a = torch.permute(tr[0], (1, 0, 2))[pos[0]]
    for i in range(1, len(tr)):
        a = torch.mm(a, torch.permute(tr[i], (1, 0, 2))[pos[i]])
    return torch.trace(a)

def _tr_recover(tr):
    shape = [t.shape[0] for t in tr]
    t_shape = [t.shape[1] for t in tr]
    rec = torch.zeros(*t_shape)
    for x in itertools.product(*([range(k) for k in shape])):
        r = tr[0][x[0], :, x[1]]
        for i in range(1, len(tr)):
            v = tr[i][x[i], :, x[(i + 1) % len(tr)]]
            #print("v:",v[*([None] * i), :].shape)
            #print("r:",r.unsqueeze(-1).shape)
            r = v[*([None] * i), :] * r.unsqueeze(-1) 
            #print("final:", r.shape)
        rec += r
    return rec

class TR():
    # té els cores de la TR decomp
    def __init__(self, source, shape, rank):
        # factors = _pFactors(N)
        print("Source:", source)
        tr_shape = _pFactors(np.prod(shape))
        print(tr_shape)

def randn(rank, shape):
    return TR(torch.rand(shape), shape, rank)