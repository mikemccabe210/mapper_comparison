## Spectral differences

import numpy as np
import networkx as nx

def unweighted_lambda_laplacian_diff(net1, net2):
    return weighted_lambda_laplacian_diff(net1, net2, weighted=None)


def unweighted_lambda_normalized_laplacian_diff(net1, net2):
    return weighted_lambda_normalized_laplacian_diff(net1, net2, weighted=None)


def unweighted_lambda_adjacency_diff(net1, net2):
    return weighted_lambda_adjacency_diff(net1, net2, weighted=None)


def unweighted_lambda_normalized_adjacency_diff(net1, net2):
    return weighted_lambda_normalized_adjacency_diff(net1, net2, weighted=None)


def unweighted_lambda_normalized_laplacian_diff(net1, net2):
    return weighted_lambda_normalized_laplacian_diff(net1, net2, weighted=None)


def weighted_lambda_laplacian_diff(net1, net2, weighted='weight'):
    """
    """
    net1_eigs = np.linalg.eigvalsh(nx.laplacian_matrix(net1, weight=weighted).toarray())
    net2_eigs = np.linalg.eigvalsh(nx.laplacian_matrix(net2, weight=weighted).toarray())
    return lambda_diff(net1_eigs, net2_eigs)


def weighted_lambda_normalized_laplacian_diff(net1, net2, weighted='weight'):
    """
    """
    net1_eigs = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(net1, weight=weighted).toarray())
    net2_eigs = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(net2, weight=weighted).toarray())
    return lambda_diff(net1_eigs, net2_eigs)


def weighted_lambda_adjacency_diff(net1, net2, weighted='weight'):
    """
    """
    net1_eigs = np.linalg.eigvalsh(nx.adj_matrix(net1, weight=weighted).toarray())
    net2_eigs = np.linalg.eigvalsh(nx.adj_matrix(net2, weight=weighted).toarray())
    return lambda_diff(net1_eigs, net2_eigs)


def weighted_lambda_normalized_adjacency_diff(net1, net2, weighted='weight'):
    adj1 = nx.adj_matrix(net1, weight=weighted).toarray()
    d1 = np.diag(1 / np.sqrt(adj1.sum(axis=1)))
    adj1 = d1.dot(adj1).dot(d1)
    adj1[np.isnan(adj1)] = 0
    adj2 = nx.adj_matrix(net2, weight=weighted).toarray()
    d2 = np.diag(1 / np.sqrt(adj2.sum(axis=1)))
    adj2 = d2.dot(adj2).dot(d2)
    adj2[np.isnan(adj2)] = 0

    net1_eigs = np.linalg.eigvalsh(adj1)
    net2_eigs = np.linalg.eigvalsh(adj2)
    return lambda_diff(net1_eigs, net2_eigs)


def lambda_diff(eigs1, eigs2):
    eigs1 = -eigs1
    eigs1.sort()
    eigs1 = -eigs1

    eigs2 = -eigs2
    eigs2.sort()
    eigs2 = -eigs2

    len1 = eigs1.shape[0]
    len2 = eigs2.shape[0]
    diff = len1 - len2
    if diff > 0:
        eigs2 = np.concatenate([eigs2, np.zeros(diff)])
    elif diff < 0:
        eigs1 = np.concatenate([eigs1, np.zeros(-diff)])

    return np.sqrt(np.sum((eigs1 - eigs2) ** 2))


def get_adjacency(net):
    adj = nx.adjacency_matrix(net).toarray()
    np.fill_diagonal(adj, 0)
    return adj