import numpy as np
import torch

class Graph:
    """
    The Graph to model the skeletons extracted by the openpose
    
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For EfficientGCN, we usually use 'spatial' or 'uniform'. 
        Last v2 uses 'spatial' (3 subsets: self, centripetal, centrifugal).
        
        layout (string): must be one of the follow candidates
        - ntu-rgb+d: Is consist of 25 joints. For more information, please
        refer to https://github.com/shahroudy/NTURGB-D
    """
    def __init__(self, layout='ntu-rgb+d', strategy='spatial', max_hop=1, dilation=1, raw_partitions=False):
        self.max_hop = max_hop
        self.dilation = dilation
        self.raw_partitions = raw_partitions

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                             (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                             (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                             (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            # Correcting 1-based to 0-based
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Do not support this layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        # For v3: use raw 0/1 adjacency for partitioning so that
        # normalize_symdigraph can be applied exactly once downstream.
        # For v1/v2 backward compat: default uses normalize_digraph output.
        partition_src = adjacency if self.raw_partitions else normalize_adjacency

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = partition_src
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = partition_src[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = partition_src[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = partition_src[j, i]
                            else:
                                a_further[j, i] = partition_src[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do not support this strategy.")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_symdigraph(A):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}.

    Better gradient flow than D^{-1}A (asymmetric) because both
    sides of the adjacency are scaled equally.  Used by LAST-E v3.
    """
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn_half = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn_half[i, i] = Dl[i] ** (-0.5)
    return np.dot(np.dot(Dn_half, A), Dn_half)


def normalize_symdigraph_full(A_subsets):
    """Symmetric normalization using FULL graph degree for all subsets.

    Instead of computing D per-subset (which over-weights self-loops and
    under-weights sparse subsets), this computes the degree from the
    combined adjacency (sum of all subsets) and applies it uniformly:

        D_full = diag(sum_k A_k · 1)
        A_norm[k] = D_full^{-1/2} · A_k · D_full^{-1/2}

    This gives balanced weights: a self-loop at spine (degree 9) gets weight
    1/9 ≈ 0.11 (correct), not 1.0 (per-subset bug).

    Args:
        A_subsets: numpy array (K, V, V) — raw 0/1 adjacency subsets.

    Returns:
        A_norm: numpy array (K, V, V) — normalized subsets.
    """
    K, V, _ = A_subsets.shape
    A_full = A_subsets.sum(axis=0)            # (V, V)
    Dl = np.sum(A_full, 0)                    # (V,)
    Dn_half = np.zeros((V, V))
    for i in range(V):
        if Dl[i] > 0:
            Dn_half[i, i] = Dl[i] ** (-0.5)
    return np.stack([
        np.dot(np.dot(Dn_half, A_subsets[k]), Dn_half)
        for k in range(K)
    ])

