"""
Contains base distances between nodes, namely:

Hausdorff distance (the maximum closest point distance)
Intersection distance (Minimum point to point distance - probably has a formal name)
Jaccard - for nodes that have perfect overlap, we can examine their Jaccard coeficient
"""

import numpy as np
from numba import jit


def node_intersection_distance(node1, node2, metric_space):
    """ Returns the minimum distance between points in a pair of
    subsets.

    """
    if node1.intersection(node2):
        return 0

    node1 = list(node1)
    node2 = list(node2)
    sub_dist = metric_space[node1][:, node2]

    return sub_dist.min()


def node_merge_distance(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.

    This is not a true distance as it is neither symmetric nor
    has an identity property.


    """
    node_diff = node2.difference(node1)
    if not node_diff:
        return 0
    node1 = list(node1)
    node2 = list(node_diff)
    sub_dist = metric_space[node1][:, node2]

    return sub_dist.min(axis = 0).max()

def node_merge_distance_matrix(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.

    This is basically a Hausdorf distance.

    """
    node_diff = node2.difference(node1)
    if not node_diff:
        return np.array([0])
    node1 = list(node1)
    node2 = list(node_diff)
    #     print len(node1), len(node2)
    sub_dist = metric_space[node1][:, node2]

    return sub_dist

def node_mutual_merge_distance(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.

    This is basically a Hausdorf distance.

    """
    node_dists = node_merge_distance_matrix(node1, node2, metric_space)
    if 1 in node_dists.shape:
        #         print node_dists.shape
        return node_dists.max()
    return max(node_dists.min(axis = 0).max(), node_dists.min(axis = 1).max())



def separation(node1, metric_space):
    """ Returns the separation of the subset of the metric
    space defined by node1 (the minimum non-zeroseparating space)
    """
    if len(node1) <= 1:
        return 0
    node1 = list(node1)
    sub_dist = metric_space[node1][:, node1]
    sub_dist[sub_dist == 0] = np.inf
    sdmin = sub_dist.min()
    if sdmin == np.inf:
        return 0

    return sdmin


def diameter(node1, metric_space):
    """ Returns the diameter of the subset of the metric space
    defined by node1 (the maximum distance)
    """
    if len(node1) <= 1:
        return 0
    node1 = list(node1)
    sub_dist = metric_space[node1][:, node1]
    np.fill_diagonal(sub_dist, -np.inf)
    return sub_dist.max()

def normalized_network_merge_distance(net1, net2, metric_space):
    """
    """
    mutual_merge_dists = network_merge_distance(net1, net2, metric_space)
    mutual_merge_dist = max(mutual_merge_dists.min(axis = 0).max(), mutual_merge_dists.min(axis = 1).max())
    return mutual_merge_dist #/ diameter(set(range(metric_space.shape[0])), metric_space)

def nodewise_jaccard(net1, net2):
    inter = net1.node_row_matrix.toarray().dot(net2.node_row_matrix.toarray().T)
    d1 = np.diag(net1.adjacency_matrix.toarray())
    d2 = np.diag(net2.adjacency_matrix.toarray())
    union = np.add.outer(d1, d2) - inter
    return inter/union


def network_merge_distance(net1, net2, metric_space):
    """
    """
    net_clust1 = net1.export_clustering_as_cover()
    net_clust2 = net2.export_clustering_as_cover()
    #     net1_part_list = [list(p.members_) for p in net_clust1.partitions_]
    #     net2_part_list = [list(p.members_) for p in net_clust2.partitions_]
    #     print net1_part_list
    #     return _net_merge_internal(net1_part_list, net2_part_list, metric_space)
    n = len(net_clust1.partitions_)
    m = len(net_clust2.partitions_)

    merge_dists = np.zeros((n, m))

    for i, p1 in enumerate(net_clust1.partitions_):
        for j, p2 in enumerate(net_clust2.partitions_):
            merge_dists[i, j] = node_mutual_merge_distance_opt(p1.members_, p2.members_, metric_space)

    return merge_dists


@jit(nopython=True)
def _net_merge_internal(net1, net2, metric_space):
    #     return 0
    n = len(net1)
    m = len(net2)
    merge_dists = np.zeros((n, m))
    for i, p1 in enumerate(net1):
        for j, p2 in enumerate(net2):
            merge_dists[i, j] = node_mutual_merge_distance_opt(set(p1), set(p2), metric_space)
    return merge_dists


@jit(nopython=True)
def node_merge_distances_opt(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.

    This is basically a Hausdorf distance.

    """
    node_diff = node2.difference(node1)
    if len(node_diff) == 0:
        return 0.0
    max_val = 0.0
    for n2 in node_diff:
        min_val = np.inf
        for n1 in node1:
            min_val = min(min_val, metric_space[n1, n2])
        max_val = max(max_val, min_val)
    return max_val  # .max()


@jit(nopython=True)
def node_mutual_merge_distance_opt(node1, node2, metric_space):
    """ Returns the size of the neighborhood expansion that would
    be required for node1 to subsume node2.

    This is basically a Hausdorf distance.
    """
    # node_dists = node_merge_distances_opt(node1, node2, metric_space)
    return max(node_merge_distances_opt(node1, node2, metric_space),
               node_merge_distances_opt(node2, node1, metric_space))