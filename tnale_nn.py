from tn import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import uuid
import itertools
import copy
import math
import time
import random

class TNLinearLayer(nn.Module):
    def __init__(self, in_shape, out_shape, G, R, bias: bool = True, initial_training = None):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layer = TN(G, in_shape + out_shape, R)
        self.n_info = {}
        self.bias = bias
        self.dim_x = np.prod(in_shape)
        self.dim_y = np.prod(out_shape)
        self.n_info["ori_params"] = self.dim_x * self.dim_y
        self.n_info["t_params"] = self.layer.get_tn_size()
        self.in_shape = in_shape
        self.G = G
        self.R = R

        self.bias_param = nn.Parameter(torch.randn(self.dim_y))

        # print("CR: " + str(self.n_info["ori_params"] / self.n_info["t_params"]))
        if self.bias:
            self.bias_param = nn.Parameter(torch.randn(self.dim_y))
        if initial_training is not None:
            # Initial training té un tensor
            TN.als_grad(self.layer, initial_training, 1e-9, iter_num=200, print_iters=True)
    def forward(self, x):
        r = self.layer(torch.reshape(x, (x.numel() // self.dim_x, self.dim_x)))
        r += self.bias_param
        return r
    def get_cr(self):
        return self.layer.get_tn_size()

class LeNetTN(nn.Module):
    def __init__(self, name, Gs, Rs, Ss):
        super().__init__()
        self.n_info = {}
        self.flatten = nn.Flatten()
        self._name = name

        #G1 = nx.Graph()
        #G2 = nx.Graph()
        #G3 = nx.Graph()
        #G1.add_edges_from([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1]])
        #G2.add_edges_from([[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]])
        #G3.add_edges_from([[1, 2], [2, 3], [3, 1]])
        
        #self.l1 = TNLinearLayer([7,7,4,4], [4,9,5], G1, [14,14,14,14,14,14,14], bias=True)
        #self.l2 = TNLinearLayer([4, 9, 5], [10, 10], G2, [10,10,10,10,10], bias=True)
        #self.l3 = TNLinearLayer([10,10], [10], G3, [7,7,7], bias=True)

        self.l1 = TNLinearLayer(Ss[0], Ss[1], Gs[0], Rs[0], bias=True)
        self.l2 = TNLinearLayer(Ss[1], Ss[2], Gs[1], Rs[1], bias=True)
        self.l3 = TNLinearLayer(Ss[2], Ss[3], Gs[2], Rs[2], bias=True)
        self.classifier = nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3, # Returns logits
        )
        self.n_info["ori_params"] = sum([x.n_info["ori_params"] for x in [self.l1, self.l2, self.l3]])
        self.n_info["t_params"] = sum([x.n_info["t_params"] for x in [self.l1, self.l2, self.l3]])
        self.n_info["cr"] = self.n_info["ori_params"] / self.n_info["t_params"]

        #print("LeNet TN ---")
        #print("Original params: " + str(self.n_info["ori_params"]))
        #print("TN params: " + str(self.n_info["t_params"]))
        #print("Compression ratio: " + str(self.n_info["cr"]))

    def get_cr(self):
        return sum([x.n_info["ori_params"] for x in [self.l1, self.l2, self.l3]]) / sum([x.get_cr() for x in [self.l1, self.l2, self.l3]])
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self._name
































        