from __future__ import division

import numpy as np

from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse.csgraph._validation import validate_graph
from sklearn.utils import check_array

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances


class SingleLinkageHistogram(BaseEstimator, ClusterMixin):
    """SingleLinkageClustering with cut-off distance determined by histogram method

    Borrows large chunks from: https://github.com/jakevdp/mst_clustering 
    for efficient MST framework

    Parameters
    ------------
    approximate : bool, optional (default: True)
       If True, then compute the approximate minimum spanning tree using
        n_neighbors nearest neighbors. If False, then compute the full
        O[N^2] edges (see Notes, below).
    n_neighbors : int, optional (default: 20)
        maximum number of neighbors of each point used for approximate
        Euclidean minimum spanning tree (MST) algorithm.  Referenced only
        if ``approximate`` is False. See Notes below.
    metric : string (default "euclidean")
        Distance metric to use in computing distances. If "precomputed", then
        input is a [n_samples, n_samples] matrix of pairwise distances (either
        sparse, or dense with NaN/inf indicating missing edges)
    metric_params : dict or None (optional)
        dictionary of parameters passed to the metric. See documentation of
        sklearn.neighbors.NearestNeighbors for details.

    Attributes
    ----------
    full_tree_ : sparse array, shape (n_samples, n_samples)
        Full minimum spanning tree over the fit data
    T_trunc_ : sparse array, shape (n_samples, n_samples)
        Non-connected graph over the final clusters
    labels_: array, length n_samples
        Labels of each point

    """

    def __init__(self, approximate=True, n_neighbors=20,
                 threshold='histogram',
                 metric='euclidean', metric_params=None):
        self.approximate = approximate
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        threshold : str
            Algorithm to use for thresholding edge length in MST
        """
        # Compute the distance-based graph G from the points in X
        if self.metric == 'precomputed':
            # Input is already a graph. Copy if sparse
            # so we can overwrite for efficiency below.
            self.X_fit_ = None
            G = validate_graph(X, directed=True,
                               csr_output=True, dense_output=False,
                               copy_if_sparse=True, null_value_in=np.inf)
        elif not self.approximate:
            X = check_array(X)
            self.X_fit_ = X
            kwds = self.metric_params or {}
            G = pairwise_distances(X, metric=self.metric, **kwds)
            G = validate_graph(G, directed=True,
                               csr_output=True, dense_output=False,
                               copy_if_sparse=True, null_value_in=np.inf)
        else:
            # generate a sparse graph using n_neighbors of each point
            X = check_array(X)
            self.X_fit_ = X
            n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
            G = kneighbors_graph(X, n_neighbors=n_neighbors,
                                 mode='distance',
                                 metric=self.metric,
                                 metric_params=self.metric_params)

        # HACK to keep explicit zeros (minimum spanning tree removes them)
        zero_fillin = G.data[G.data > 0].min() * 1E-8
        G.data[G.data == 0] = zero_fillin

        # Compute the minimum spanning tree of this graph
        self.full_tree_ = minimum_spanning_tree(G, overwrite=True)

        # undo the hack to bring back explicit zeros
        self.full_tree_[self.full_tree_ == zero_fillin] = 0

        if self.threshold == 'hermite':
            max_edge = self._hermite_threshold(self.full_tree_)
        else:
            max_edge = self._histogram_threshold(self.full_tree_)

        mask = self.full_tree_.data > max_edge

        cluster_graph = self.full_tree_.copy()
        original_data = cluster_graph.data
        cluster_graph.data = np.arange(1, len(cluster_graph.data) + 1)
        cluster_graph.data[mask] = 0
        cluster_graph.eliminate_zeros()
        cluster_graph.data = original_data[cluster_graph.data.astype(int) - 1]

        self.cluster_graph_ = cluster_graph
        self.n_components_, self.labels_ = connected_components(cluster_graph,
                                                                directed=False)
        return self

    def _histogram_threshold(self, mst):
        """
        Implements thresholding based on creating a histogram of point to
        point distances. The first "gap" in the histogram is used as the
        threshold

        Parameters
        ----------
        mst - scipy.sparse - matrix containing the distances on the MST

        Returns
        -------
        threshold - float - float value indicating maximum edge length for the clustering algorithm
        """
        eps = 1e-7
        counts, mins = np.histogram(mst.data, bins='auto')
        first_empty = np.where(counts == 0)[0]
        if first_empty.any():
            first_empty = first_empty[0]
            max_edge = mins[first_empty] + eps
        # print max_edge
        else:
            max_edge = mst.data.max() + eps
        # print 'nocut'

        return max_edge

    def _hermite_threshold(self, mst):
        """
        Implements thresholding based on the hermite function. Attempts
        to find a split point by identifying maximum values in the hermite function.

        Parameters
        ----------
        mst - scipy.sparse - matrix containing the distances on the MST

        Returns
        -------
        threshold - float - float value indicating maximum edge length for the clustering algorithm
        """
        lengths = np.sort(mst.data)
        mn = lengths[0]
        mx = lengths[-1]

        num_range = mx - mn
        sigma = num_range / 50
        delta = num_range * 0.1
        threshold = mx + delta

        resolution = 500
        incr = num_range / resolution

        if num_range == 0:
            return threshold

        lastVal = 0.0

        x = mn
        while x <= 2 * mx:
            total_sum = 0.0
            for dist in lengths:
                xml = x - dist
                total_sum += -2 * xml * np.exp(-xml * xml / sigma) / sigma
            if lastVal < 0 and total_sum > 0:
                threshold = x - incr / 2
                break
            lastVal = total_sum
            x += incr
        return threshold

class NotClusterer(BaseEstimator, ClusterMixin):
    """
    Dummy class used to run Mapper with no clustering step.
    """
    def fit(self, X, y=None):
        self.labels_ = np.zeros(X.shape[0])
