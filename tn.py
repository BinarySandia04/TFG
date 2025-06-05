import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import uuid
import itertools
import copy
import math
import time
import random
from PIL import Image
from torchvision import transforms
from itertools import combinations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# min_from -> x_0, max_from -> x_1
# min_to -> y_0, max_to -> y_1
# x_0, x_1, y_0, y_1
def lerp(min_from, max_from, min_to, max_to, value):
    if max_from == min_from:
        return min_to
    return min_to + (value - min_from) * (max_to - min_to) / (max_from - min_from)

def als_solve(A, B):
    return torch.linalg.lstsq(A, B).solution
def unfold(tensor, mode):
    """
    Unfolds an n-th order tensor along the specified mode.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (d0, d1, ..., dn)
        mode (int): Mode along which to unfold (0-based)
        
    Returns:
        torch.Tensor: The mode-i unfolding of shape (tensor.size(mode), -1)
    """
    # Move the mode to the first dimension
    new_order = [mode] + [i for i in range(tensor.ndim) if i != mode]
    permuted = tensor.permute(new_order)
    
    # Flatten all dimensions except the first (which is mode)
    unfolded = permuted.reshape(tensor.shape[mode], -1)
    return unfolded
def fold(unfolded, mode, shape):
    """
    Refolds a matrix back into a tensor of given shape along a mode.
    
    Args:
        unfolded (torch.Tensor): Unfolded matrix of shape (shape[mode], -1)
        mode (int): Mode along which it was unfolded (0-based)
        shape (tuple): Original shape of the tensor before unfolding
        
    Returns:
        torch.Tensor: The refolded tensor of shape `shape`
    """
    # Calculate the shape after permuting mode to front
    new_order = [mode] + [i for i in range(len(shape)) if i != mode]
    inverse_order = list(torch.argsort(torch.tensor(new_order)))

    # Compute the shape of the permuted tensor
    permuted_shape = (shape[mode], -1)
    reshaped = unfolded.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode])

    # Invert the permutation to get original order
    folded = reshaped.permute(*inverse_order)
    return folded
def unfold_to_matrix(x: torch.Tensor, dims_to_rows: list[int]) -> torch.Tensor:
    """
    Unfolds a tensor into a matrix, flattening `dims_to_rows` into rows,
    and the rest into columns.

    Args:
        x (torch.Tensor): The input tensor.
        dims_to_rows (list[int]): Dimensions to flatten into the row axis.

    Returns:
        torch.Tensor: A 2D tensor (matrix) of shape (prod(dims_to_rows), prod(other_dims)).
    """
    all_dims = list(range(x.ndim))
    dims_to_cols = [d for d in all_dims if d not in dims_to_rows]

    # Permute to bring row dims first, then col dims
    permuted_dims = dims_to_rows + dims_to_cols
    x_permuted = x.permute(permuted_dims)

    # Compute new shape
    row_size = int(torch.prod(torch.tensor([x.shape[d] for d in dims_to_rows])))
    col_size = int(torch.prod(torch.tensor([x.shape[d] for d in dims_to_cols])))

    return x_permuted.reshape(row_size, col_size)
def get_inverse_perm(p):
    r = []
    for i in range(len(p)):
        # Busquem i en p, la seva pos s'afegeix a r
        for j in range(len(p)):
            if p[j] == i:
                r.append(j)
                break
    return r
def find_permutation(X, Y):
    """
    Returns a list of indices such that applying this permutation to X results in Y.
    Works with unhashable elements.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must be of the same length.")

    used = [False] * len(Y)
    permutation = []

    for x in X:
        found = False
        for i, y in enumerate(Y):
            if not used[i] and x == y:
                permutation.append(i)
                used[i] = True
                found = True
                break
        if not found:
            raise ValueError("Y is not a permutation of X.")

    return permutation

#ef graph_neighborhood(G):
#    nodes = list(G.nodes())
#
#    for u, v in combinations(nodes, 2):
#        mapping = {u: v, v: u}  # swap u and v
#        G_swapped = nx.relabel_nodes(G, mapping, copy=True)
#        yield G_swapped


def graph_neighborhood(G):
    """
    Generator that yields non-isomorphic graphs obtained by adding or removing
    exactly one edge from G.
    
    Parameters:
        G (networkx.Graph): The original graph.
    
    Yields:
        networkx.Graph: A new graph with one edge added or removed.
    """
    seen = set()
    nodes = list(G.nodes())
    existing_edges = set(G.edges())
    l = len(existing_edges)

    def canonical_form(H):
        # You could use: nx.to_graph6_bytes(H) for a hashable canonical label
        return nx.to_graph6_bytes(H, header=False)  # Compact & canonical

    i = 0
    # 1. Remove one edge at a time
    for u, v in existing_edges:
        H = G.copy()
        H.remove_edge(u, v)
        key = canonical_form(H)
        if key not in seen:
            seen.add(key)
            yield (H, i, True)
        i += 1
    
    # 2. Add one non-existing edge at a time
    for u, v in combinations(nodes, 2):
        if (u, v) not in existing_edges and (v, u) not in existing_edges:
            H = G.copy()
            H.add_edge(u, v)
            key = canonical_form(H)
            if key not in seen:
                seen.add(key)
                yield (H, i, False)

class TN(torch.nn.Module):
    def initTensor(self, shape):
        return torch.randn(shape, device=device)
    def __init__(self, G, sizes, ranks):
        super().__init__()
        if len(G.edges) != len(ranks):
            raise Exception("Rank length must be equal to the number of edges")
        if len(G.nodes) != len(sizes):
            raise Exception("Sizes length must be equal to the number of nodes")
        self.data = {}
        self.tensors = {}
        params_dict = {}

        self.total_size = 1
        # Init data
        i = 0
        for node in G.nodes:
            self.data[node] = [ [sizes[i], -1, i] ]
            self.total_size *= sizes[i]
            i += 1
        i = 0
        for e in G.edges:
            node1, node2 = (e[0], e[1])
            h = uuid.uuid4()
            self.data[node1] += [  [ranks[i], h] ]
            self.data[node2] += [  [ranks[i], h]  ]
            i += 1
        i = 0
        for node in G.nodes:
            shape = []
            for x in self.data[node]:
                shape.append(x[0])
            self.tensors[node] = self.initTensor(shape)
            params_dict[str(node)] = torch.nn.Parameter(self.initTensor(shape))
            i += 1
        self.params = torch.nn.ParameterDict(params_dict)
        # print(self.data)
        # print(self.tensors)
    def set_core(self, tensor, node):
        self.tensors[node] = tensor
        #with torch.no_grad():
        #    self.params[node] = tensor
        
    def get_core(self, node):
        return self.tensors[node]
    def get_tn_size(self):
        s = 0
        for k in self.tensors.keys():
            s += torch.numel(self.tensors[k])
        return s
    def contract(self, node1, node2, newnode, node_data, tensor_data):
        # Compute other indexes
        dim1 = []
        dim2 = []
        inode1 = int(node1)
        inode2 = int(node2)
        # print(node_data)
        for i in range(len(node_data[inode1])):
            for j in range(len(node_data[inode2])):
                if node_data[inode1][i][1] == -1 or node_data[inode2][j][1] == -1:
                    continue
                if node_data[inode1][i][1] == node_data[inode2][j][1]:
                    dim1.append(i)
                    dim2.append(j)

        
        ts = []
        for i in range(len(node_data[inode1])):
            if i not in dim1:
                ts.append(node_data[inode1][i])
        for i in range(len(node_data[inode2])):
            if i not in dim2:
                ts.append(node_data[inode2][i])
        # Compute
        # print(tensor_data)
        # print(tensor_data[node1])
        t = torch.tensordot(tensor_data[node1], tensor_data[node2], dims=(dim1, dim2))
        node_data[int(newnode)] = ts
        tensor_data[newnode] = t

        del node_data[inode1]
        del node_data[inode2]
        del tensor_data[node1]
        del tensor_data[node2]

    def forward(self, x):
        # TODO
        # Volem multiplicar x pel tensor de avaluar la net sencera!!!
        r = x.shape[1]
        return x.matmul(TN.eval_params(self).reshape(r, self.total_size // r))

    
    """
    Performs Alternating Local Enumeration
    
    Args:
        G0 (nx.Graph): Starting graph
        R0 (int[]): Starting ranks
        radius (int): Search radius of ranks
        iters (int): Number of iterations
        
    Returns:
        (G, R): The optimal G and R
    """
    @staticmethod
    def tn_ale(G0, R0, radius, iters, objective, print_iters=False, max_rank=20, tuning_param=0.5, update_graph=False):
        G, R = (G0, R0)
        h = TN.evaluate_structure(G, R, objective, tuning_param=tuning_param)
        p = (G, R)
        for d in range(iters):
            # First rank update
            for k in range(len(R)):
                minimum, maximum = TN.evaluate_structure_interpolated(G, R, k, radius, objective, tuning_param=tuning_param, max_rank=max_rank) 
                for i in range(-radius, radius):
                    Rp = copy.deepcopy(R)
                    Rp[k] += i
                    if Rp[k] < 0:
                        Rp[k] = 0
                    elif Rp[k] > max_rank:
                        Rp[k] = max_rank
                        
                    if Rp[k] < R[k]:
                        evaluation = lerp(max(0, R[k] - radius), R[k], minimum, h, Rp[k])
                    else:
                        evaluation = lerp(R[k], min(R[k] + radius, max_rank), h, maximum, Rp[k])
                    
                    if h > evaluation:
                        p = (G, Rp)
                        h = evaluation
                G, R = p
            # We now update G
            if update_graph:
                for nG, index, rem in graph_neighborhood(G):
                    if rem:
                        Rp = copy.deepcopy(R)
                        del Rp[index]
                        evaluation = TN.evaluate_structure(nG, Rp, objective, tuning_param=tuning_param)
                        if h > evaluation:
                            p = (nG, Rp)
                            h = evaluation
                    else:
                        for x in [1, radius//2, radius]:
                            Rp = copy.deepcopy(R)
                            Rp.insert(index, x)
                            evaluation = TN.evaluate_structure(nG, Rp, objective, tuning_param=tuning_param)
                            if h > evaluation:
                                p = (nG, Rp)
                                h = evaluation
                G, R = p
            # Second rank update
            for k in range(len(R) - 1, 0, -1):
                minimum, maximum = TN.evaluate_structure_interpolated(G, R, k, radius, objective, tuning_param=tuning_param, max_rank=max_rank)
                for i in range(-radius, radius):
                    Rp = copy.deepcopy(R)
                    Rp[k] += i
                    if Rp[k] < 0:
                        Rp[k] = 0
                    elif Rp[k] > max_rank:
                        Rp[k] = max_rank
                        
                    if Rp[k] < R[k]:
                        evaluation = lerp(max(0, R[k] - radius), R[k], minimum, h, Rp[k])
                    else:
                        evaluation = lerp(R[k], min(R[k] + radius, max_rank), h, maximum, Rp[k])
                        
                    if h > evaluation:
                        p = (G, Rp)
                        h = evaluation
                G, R = p
            if print_iters:
                print("Iter " + str(d) + " done")
                print("R: " + str(R))
                print("G edges: " + str(G.edges()))
        return (G, R)

    
    @staticmethod
    def evaluate_structure_interpolated(G, R, k, radius, objective, tuning_param=2, max_rank=25):
        Rp = copy.deepcopy(R)
        Rp[k] -= radius
        if Rp[k] < 0:
            Rp[k] = 0
        minimum = TN.evaluate_structure(G, Rp, objective, tuning_param=tuning_param)
        Rp[k] = R[k] + radius
        if Rp[k] > max_rank:
            Rp[k] = max_rank
        maximum = TN.evaluate_structure(G, Rp, objective, tuning_param=tuning_param)
        return (minimum, maximum)
    
    # Higher tuning_param leads to less relative error at the cost of less compression
    @staticmethod
    def evaluate_structure(G, R, objective, tuning_param=2):
        # We do one iteration of ALS):
        tn = TN(G, objective.shape, R)
        graph = TN.als(tn, objective, 0, iter_num=5)[1]
        icr = tn.get_tn_size() / objective.numel()
        return icr + min(graph[1]) * tuning_param
    
    @staticmethod
    def eval(t):
        n_data = copy.copy(t.data)
        t_data = copy.copy(t.tensors)
        
        p = len(n_data) + 1
        while len(n_data) > 1:
            klist = list(n_data.keys())
            x = klist[0]
            y = klist[1]
            t.contract(x, y, p, node_data=n_data, tensor_data=t_data)
            p += 1
        # TODO: Fer reshape al return ja que les dims estàn cambiades!
        perm = [[y[2] for y in n_data[p-1]].index(x) for x in range(len(n_data[p-1]))]
        return torch.permute(t_data[p-1], perm)

    @staticmethod
    def eval_params(t):
        n_data = copy.copy(t.data)
        t_data = dict(t.params)
        
        p = len(n_data) + 1
        while len(n_data) > 1:
            klist = list(n_data.keys())
            x = klist[0]
            y = klist[1]
            t.contract(str(x), str(y), str(p), node_data=n_data, tensor_data=t_data)
            p += 1
        perm = [[y[2] for y in n_data[p-1]].index(x) for x in range(len(n_data[p-1]))]
        return torch.permute(t_data[str(p-1)], perm)
    
    @staticmethod
    def get_contraction_except(tn, core):
        n_data = copy.copy(tn.data)
        t_data = copy.copy(tn.tensors)
        p = len(n_data) + 1
        while len(n_data) > 2:
            klist = list(n_data.keys())
            klist.sort()
            e = [klist[0], klist[1], klist[2]]
            if core in e:
                e.remove(core)
            
            x = e[0]
            y = e[1]
            
            tn.contract(x, y, p, node_data=n_data, tensor_data=t_data)
            p += 1
        
        return (t_data[p-1], t_data[core], n_data[p-1], n_data[core])
    
    @staticmethod
    def als(tn, t, err, iter_num=float('inf'), print_iters=False, tikhonov=False, max_time=float('inf')):
        lambda_reg = 1e-5
        iters = 1
        rel_err = torch.norm(TN.eval(tn) - t).item() / torch.norm(t).item()

        time_resh = 0
        x = []
        y = []
        start_time = time.time()
        
        while rel_err >= err:
            rel_err = torch.norm(TN.eval(tn) - t).item() / torch.norm(t).item()
            # print(torch.norm(TN.eval(tn) - t))
            x.append(iters)
            y.append(rel_err)
            if print_iters and (iters % 10 == 0 or iter_num < 30):
                if iter_num < 300 or iters % 100 == 0:
                    print("epoch " + str(iters) + "/" + str(iter_num) + " err: " + str(rel_err))
                    
            if iters > iter_num:
                break
            for k in range(1, len(tn.tensors) + 1):
                # Aqui hem de fer el reshape
                cont_ten, orig_ten, cont_data, orig_data = TN.get_contraction_except(tn, k)

                uuid_order = []
                for p in orig_data:
                    if p[1] != -1:
                        uuid_order.append(p[1])

                # Volem una permutació que deixi en uuid_order els edges de cont_ten
                perm = [-1] * len(cont_ten.shape)
                tperm = []
                i = 0
                nx = 1
                for j in range(len(cont_data)):
                    if cont_data[j][1] == -1:
                        perm[j] = i
                        tperm.append(cont_data[j][2])
                        i += 1
                        nx *= cont_data[j][0]
                ny = 1
                for u in uuid_order:
                    for j in range(len(cont_data)):
                        if cont_data[j][1] == u:
                            perm[j] = i
                            i += 1
                            ny *= cont_data[j][0]
                tperm.append(orig_data[0][2])
                # Tenim la permutació a perm
                # I tperm es la permutació que cal de modificar el tensor objectiu

                cont_perm = torch.permute(cont_ten, get_inverse_perm(perm))
                # I podem fer ja la matriu

                cont_mat = torch.reshape(cont_perm, (nx, ny))

                # Permutem ojectiu
                t_obj = torch.permute(t, tperm)
                
                # I fem reshape!
                obj_mat = torch.reshape(t_obj, (nx, orig_data[0][0]))
                
                # core = torch.linalg.lstsq(cont_mat, obj_mat)[0]
                
                if time.time() - start_time > max_time:
                    return (rel_err, (x, y))
                #ATA = cont_mat.T @ cont_mat + lambda_reg * torch.eye(cont_mat.shape[1], device=cont_mat.device, dtype=cont_mat.dtype)
                #ATb = cont_mat.T @ obj_mat
                #core = torch.linalg.solve(ATA, ATb)
                if tikhonov:
                    ATA = cont_mat.T @ cont_mat + lambda_reg * torch.eye(cont_mat.shape[1], device=cont_mat.device, dtype=cont_mat.dtype)
                    ATb = cont_mat.T @ obj_mat
                    core = torch.linalg.solve(ATA, ATb)
                else:
                    core = torch.linalg.pinv(cont_mat) @ obj_mat
                
                end_time = time.time()
                

                # Ara busquem el core
                shape = orig_ten.shape
                shape_left = shape[1:] + shape[:1]

                res_core = torch.reshape(core, shape_left)
                res_core = res_core.permute(res_core.ndim - 1, *range(0, res_core.ndim - 1))

                tn.set_core(res_core, k)
                time_resh += end_time - start_time
            iters += 1
            #return
        if print_iters:
            print("time_resh: " + str(time_resh))
        
        return (rel_err, (x, y))

        
    @staticmethod
    def als_grad(tn, t, iter_num=100000, epoch=100, lr=0.1, print_iters=False, max_time=float('inf')):
        rel_err = torch.norm(TN.eval(tn) - t).item() / torch.norm(t).item()

        time_resh = 0
        x = []
        y = []

        optimizer = torch.optim.Adam(tn.params.values(), lr=lr)
        
        start_time = time.time()
        for iters in range(iter_num):
            if print_iters and (iters % 10 == 0 or iter_num < 30):
                if iter_num < 300 or iters % 100 == 0:
                    print("epoch " + str(iters) + "/" + str(iter_num) + " err: " + str(rel_err))
            for i in range(epoch):
                # print(torch.norm(TN.eval(tn) - t))
                
                
                optimizer.zero_grad()
                loss = torch.norm(TN.eval_params(tn) - t)
    
                loss.backward()
                optimizer.step()
            rel_err = torch.norm(TN.eval_params(tn) - t).item() / torch.norm(t).item()
            
            x.append(iters)
            y.append(rel_err)
            
            iters += 1
            if time.time() - start_time > max_time:
                return (rel_err, (x,y))
            #return
        if print_iters:
            print("time_resh: " + str(time_resh))
        
        return (rel_err, (x, y))






















        