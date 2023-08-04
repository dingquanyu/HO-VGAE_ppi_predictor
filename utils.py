"""
Utility scripts with necessary functions
"""

import networkx as nx
import numpy as np
import logging
import torch
import pickle as pkl
import sys 
import numpy as np
import scipy.sparse as sp
logger = logging.getLogger(__file__)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_kipf_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("input_data/kipf_data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("input_data/kipf_data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.tocoo()
    return adj, features,graph

def obtain_all_edges(G):
    """obtain all edges from G"""
    nodelist = list(G.nodes())
    raw_edges=[e for e in G.edges]
    return_edges = []
    for e in raw_edges:
        idx1 = nodelist.index(e[0])
        idx2 = nodelist.index(e[1])
        return_edges.append((idx1,idx2))

    logger.info("finished creating all edges")
    return return_edges

def create_adj_mtx(G):
    """
    Returns an adjacency matrix with sorted nodes

    Args
    G: a networkx graph objects
    """
    nodes = sorted(G.nodes())
    adj = nx.to_numpy_array(G,nodelist=nodes)
    np.fill_diagonal(adj,1)
    return sp.coo_matrix(adj)
