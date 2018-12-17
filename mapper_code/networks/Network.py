from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
from ..partitioners import Cover, Partition

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
        for row in self.node_row_matrix:
            members = np.where(row > 0)
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
        rows = []
        cols = []
        values = []
        max_cluster = 0
        node_sets = set()
        for partition in cover.partitions_:
            sub_rows = []
            p_index = list(partition.members_)
            if len(p_index) >= 3:
                model = self._sub_cluster(metric_space[p_index, :][:, p_index], clusterer)
                labels = model.labels_ - np.min(model.labels_)
                for i, label in enumerate(labels):
                    cols.append(p_index[i])
                    sub_rows.append(label + max_cluster)
                    values.append(1)
                max_cluster = max(sub_rows) + 1
                rows.extend(sub_rows)
            elif len(p_index) > 0:
                cols.append(p_index[0])
                rows.append(max_cluster)
                values.append(1)
                max_cluster += 1

            else:
                pass

                #         print len(values), len(rows), len(cols), self.N, max_cluster, max(cols)
        node_row_matrix = coo_matrix((values, (rows, cols)), shape=(max_cluster, self.N))
        return node_row_matrix.tocsr()

    def _prune(self, adj):
        """
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