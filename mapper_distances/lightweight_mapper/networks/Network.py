"""
Used for building Mapper graphs
"""

from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
from ..networks import Cover, Partition

class MetricSpace():
    """ Basically just a distance matrix and type"""

    def __init__(self, data, metric='euclidean'):
        # build metric space as distance matrix
        self.N, self.M = data.shape
        if metric == 'precomputed':
            self.dist_matrix = data
        else:  # Compute distance
            pass


class Network():
    """ Creates topological network

    """

    def __init__(self, metric_space, cover_list, clusterer,
                 prune=True, backend='networkx'):
        # build metric space as distance matrix
        self.partition_node_map = {}
        self.N = metric_space.shape[0]
        self.cover = self.build_cover(cover_list)
        self.node_row_matrix = self.build_topological_model(metric_space, self.cover, clusterer)
        self.adjacency_matrix = self.node_row_matrix.dot(self.node_row_matrix.T)
        self.cooccurence_matrix = self.node_row_matrix.T.dot(self.node_row_matrix)
        if prune:
            pruned_node_set = self._prune(self.adjacency_matrix)
            self.raw_node_row_matrix = self.node_row_matrix
            self.raw_adjacency_matrix = self.adjacency_matrix
            self.raw_cooccurence_matrix = self.cooccurence_matrix
            partition_keys = sorted(self.partition_node_map.keys())
            re_index = 0
            new_partition_node_map = {}
            for node in partition_keys:
                if node in pruned_node_set:
                    new_partition_node_map[re_index] = self.partition_node_map[node]
                    re_index += 1
            self.partition_node_map = new_partition_node_map
            self.node_row_matrix = self.node_row_matrix[pruned_node_set, :]
            self.adjacency_matrix = self.node_row_matrix.dot(self.node_row_matrix.T)
            self.cooccurence_matrix = self.node_row_matrix.T.dot(self.node_row_matrix)

        self.graph = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

    def refresh(self, prune = True):
        """ Refreshes to deal with manual changes to node_row"""
        self.adjacency_matrix = self.node_row_matrix.dot(self.node_row_matrix.T)
        self.cooccurence_matrix = self.node_row_matrix.T.dot(self.node_row_matrix)
        if prune:
            pruned_node_set = self._prune(self.adjacency_matrix)

            self.raw_node_row_matrix = self.node_row_matrix
            self.raw_adjacency_matrix = self.adjacency_matrix
            self.raw_cooccurence_matrix = self.cooccurence_matrix
            partition_keys = sorted(self.partition_node_map.keys())
            re_index = 0
            new_partition_node_map = {}
            for node in partition_keys:
                if node in pruned_node_set:
                    new_partition_node_map[re_index] = self.partition_node_map[node]
                    re_index += 1
            self.partition_node_map = new_partition_node_map
            self.node_row_matrix = self.node_row_matrix[pruned_node_set, :]
            self.adjacency_matrix = self.node_row_matrix.dot(self.node_row_matrix.T)
            self.cooccurence_matrix = self.node_row_matrix.T.dot(self.node_row_matrix)

        self.graph = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

    def build_cover(self, cover_list):
        """ Builds cover using partitions

        Pass a list of covers of the data set.

        Parameters
        -----------
        cover_list : list[Cover]
            List of cover objects describing individual partitions of the
        """
        final_cover = cover_list[0]

        for cover in cover_list[1:]:
            final_cover = final_cover.cross(cover)

        return final_cover

    def export_clustering_as_cover(self):
        """ Exports cover built from node_row_matrix

        Returns
        -------

        """
        partition_list = []
        for i in range(self.node_row_matrix.shape[0]):
            members = np.where(self.node_row_matrix[i, :].toarray() > 0)[1]
            new_desc = [{'description': 'Copied from network'}]
            part = Partition(set(members), new_desc)
            partition_list.append(part)

        return Cover(partition_list)

    def build_topological_model(self, metric_space, cover, clusterer):
        """

        Parameters
        ----------
        metric_space
        cover
        clusterer

        Returns
        -------

        """
        self.partition_node_map = {}
        rows = []
        cols = []
        values = []
        max_cluster = 0
        node_sets = set()
        for ind, partition in enumerate(cover.partitions_):
            sub_rows = []
            p_index = list(partition.members_)
            if len(p_index) >= 3:
                model = self._sub_cluster(metric_space[p_index, :][:, p_index], clusterer)
                labels = model.labels_
                max_label = labels.max() + 1
                for i, label in enumerate(labels):
                    if label == -1:
                        cols.append(p_index[i])
                        sub_rows.append(max_label + max_cluster)
                        max_label += 1
                        values.append(1)
                        self.partition_node_map[max_label + max_cluster] = ind
                    else:
                        cols.append(p_index[i])
                        sub_rows.append(label + max_cluster)
                        values.append(1)
                        self.partition_node_map[label + max_cluster] = ind
                max_cluster = max(sub_rows) + 1
                rows.extend(sub_rows)
            elif len(p_index) > 0:
                cols.extend(p_index)
                rows.extend([max_cluster] * len(p_index))
                values.extend([1] * len(p_index))
                self.partition_node_map[max_cluster] = ind
                max_cluster += 1

            else:
                pass

                #         print len(values), len(rows), len(cols), self.N, max_cluster, max(cols)
        node_row_matrix = coo_matrix((values, (rows, cols)), shape=(max_cluster, self.N))
        return node_row_matrix.tocsr()

    def _prune(self, adj):
        """ Cleans up the graph by removing nodes that are wholly contained within
        their neighbors.
        """
        n = adj.shape[0]
        delete_nodes = set()
        diag = adj.diagonal()

        for i in range(n):
            if i not in delete_nodes:
                data_row = adj[i]
                sub = data_row.indices[(data_row.data == diag[i]) & (data_row.indices != i)]
                for j in sub:
                    if diag[j] > diag[i] and j not in delete_nodes:
                        delete_nodes.add(i)
                    elif diag[j] == diag[i] and j not in delete_nodes:
                        delete_nodes.add(j)

        use_nodes = [i for i in range(n) if i not in delete_nodes]
        return use_nodes

    def _sub_cluster(self, sub_metric_space, clusterer):
        """Performs clustering within submetric space

        """
        #         print sub_metric_space.shape
        clusterer.fit(sub_metric_space)
        return clusterer