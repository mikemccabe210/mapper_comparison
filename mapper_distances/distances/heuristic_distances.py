"""
Dedicated to a greedy alignment method that works quite well, but is difficult to reason about
"""

import numpy as np
from base_distances import *
import networkx as nx


def density_zipper_dfs(network1, network2, metric_space):
    #     """
    #     """
    # Get first node

    assignments1 = network1.node_row_matrix
    assignments2 = network2.node_row_matrix

    used_nodes1 = set()
    used_nodes2 = set()

    used_points1 = set()
    used_points2 = set()

    n1 = assignments1.shape[1]
    n2 = assignments2.shape[1]

    # Calculate node densities to get starting point
    node_density1 = assignments1.sum(axis=1) / float(n1)
    densist1 = np.argmax(node_density1)
    node_density2 = assignments2.sum(axis=1) / float(n2)
    densist2 = np.argmax(node_density2)

    net1_cover = network1.export_clustering_as_cover().partitions_
    net2_cover = network2.export_clustering_as_cover().partitions_

    netx_1 = nx.from_scipy_sparse_matrix(network1.adjacency_matrix)
    netx_2 = nx.from_scipy_sparse_matrix(network2.adjacency_matrix)

    density1 = 0.0
    density2 = 0.0
    current_node1 = None
    current_node2 = None
    point_queue1 = []
    point_queue2 = []
    # Dynamic programming, woo
    node_dist_dict = {}
    total_dist = 0

    # Get highest
    if node_density1[densist1] >= node_density2[densist2]:
        current_node1 = densist1
        used_nodes1.add(current_node1)
        used_points1.update(net1_cover[current_node1].members_)
        point_queue1.append(current_node1)
        density1 = len(used_points1) / n1
        search = 2
    else:
        current_node2 = densist2
        used_nodes2.add(current_node2)
        used_points2.update(net2_cover[current_node2].members_)
        point_queue2.append(current_node2)
        density2 = len(used_points2) / n2
        search = 1
    search_pool = set()
    added_dist = 0
    # Density zipper until all mass accounted for
    #  - possible to account for mass without completing all nodes
    while density1 + density2 < 2:
        #         print added_dist, search, 'node2', current_node2, density2, len(search_pool), 'node1', current_node1, density1, len(search_pool)
        #         print search_pool
        closest_val = np.inf
        closest_node = None
        # if density 1 is higher, search for density 2 match
        if search == 2:
            if current_node2 is None:
                search_pool = netx_2.nodes()
            else:
                search_pool = netx_2.neighbors(current_node2)
            search_pool = set(search_pool) - used_nodes2
            # if our search pool is used, find next option
            while not search_pool:
                if not point_queue2:
                    search_pool = set(netx_2.nodes()) - used_nodes2
                    if not search_pool:
                        break
                else:
                    current_node2 = point_queue2.pop()
                    search_pool = netx_2.neighbors(current_node2)
                    search_pool = set(search_pool) - used_nodes2
            for node in search_pool:
                if node not in used_nodes2:
                    node_dist = node_dist_dict.get((current_node1, node),
                                                   node_mutual_merge_distance_opt(net1_cover[current_node1].members_,
                                                                                  net2_cover[node].members_,
                                                                                  metric_space))
                    node_dist_dict[(current_node1, node)] = node_dist
                    if node_dist < closest_val:
                        closest_val = node_dist
                        closest_node = node

            current_node2 = closest_node
            point_queue2.append(current_node2)
            used_nodes2.add(current_node2)
            used_points2.update(net2_cover[current_node2].members_)
            new_dens = len(used_points2) / n2
            total_dens_diff = max(density1 - density2, 0)
            added_dist = min(new_dens - density2, total_dens_diff) * closest_val
            total_dist += added_dist
            density2 = new_dens
            if density2 >= density1:
                search = 1
        # Else use net1
        elif search == 1:
            if current_node1 is None:
                search_pool = netx_1.nodes()
            else:
                search_pool = netx_1.neighbors(current_node1)
            search_pool = set(search_pool) - used_nodes1
            # if our search pool is used, find next option
            while not search_pool:
                if not point_queue1:
                    search_pool = set(netx_1.nodes()) - used_nodes1
                    if not search_pool:
                        break
                else:
                    current_node1 = point_queue1.pop()
                    search_pool = netx_1.neighbors(current_node1)
                    search_pool = set(search_pool) - used_nodes1
            for node in search_pool:
                if node not in used_nodes1:
                    node_dist = node_dist_dict.get((node, current_node2),
                                                   node_mutual_merge_distance_opt(net1_cover[node].members_,
                                                                                  net2_cover[current_node2].members_,
                                                                                  metric_space))
                    node_dist_dict[(node, current_node2)] = node_dist
                    if node_dist < closest_val:
                        closest_val = node_dist
                        closest_node = node
            current_node1 = closest_node
            point_queue1.append(current_node1)
            used_nodes1.add(current_node1)

            used_points1.update(net1_cover[current_node1].members_)
            new_dens = len(used_points1) / n1
            total_dens_diff = max(density2 - density1, 0)
            added_dist = min(new_dens - density1, total_dens_diff) * closest_val
            total_dist += added_dist
            density1 = new_dens
            if density1 >= density2:
                search = 2
        else:
            if node_density1[densist1] >= node_density2[densist2]:
                current_node1 = densist1
                used_nodes1.add(current_node1)
                used_points1.update(net1_cover[current_node1].members_)
                density1 = len(used_points1) / n1
            else:
                current_node2 = densist2
                used_nodes2.add(current_node2)
                used_points2.update(net2_cover[current_node2].members_)
                density2 = len(used_points2) / n2

                # if density is equal, choose next point but dont add
                # find next density point in pools and add that
    return total_dist  # / ((len(used_points1) + len(used_points2))/2.0)


def normalized_density_zipper(network1, network2, metric_space):
    return density_zipper_dfs(network1, network2, metric_space) - np.sum([density_zipper_dfs(network1, network1, metric_space),
                                                                          density_zipper_dfs(network2, network2,
                                                                                     metric_space)]) / 2.0


def density_zipper_similarity(network1, network2, metric_space):
    return np.mean([density_zipper_dfs(network1, network1, metric_space),
                    density_zipper_dfs(network2, network2, metric_space)]) / density_zipper_dfs(network1, network2,
                                                                                        metric_space)
