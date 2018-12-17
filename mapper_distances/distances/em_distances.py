"""
Large set of EMD-type distances used in experimentation
"""
from __future__ import division
import numpy as np
from numba import jit, njit
from base_distances import *
import pyemd
import networkx as nx
import ot
import scipy.sparse as sparse
from ..lightweight_mapper.networks import Network, Cover, Partition
from ..lightweight_mapper.clustering import NotClusterer

from ortools.linear_solver import pywraplp


def LP_wrapper():
    pass


def make_constraints(mu1, mu2, params, solver):
    """ Makes constraints for LP"""
    constraints_list = []
    nrows = mu1.shape[0] + mu2.shape[0]
    for i in range(mu1.shape[0]):
        c = solver.Constraint(mu1[i], mu1[i])
        for j in range(mu2.shape[0]):
            c.SetCoefficient(params[i * mu2.shape[0] + j], 1)
        constraints_list.append(c)

    for j in range(mu2.shape[0]):
        c = solver.Constraint(mu2[j], mu2[j])
        for i in range(mu1.shape[0]):
            c.SetCoefficient(params[i * mu2.shape[0] + j], 1)
        constraints_list.append(c)
    return constraints_list


def initial_flb():
    pass


from numba import njit


@njit
def compute_gw_from_transport(mu, D1, D2):
    """
    """
    n = D1.shape[0]
    m = D2.shape[0]
    mu = mu.reshape((n, m))
    rval = 0
    for i in range(n):
        for ii in range(n):
            for j in range(m):
                for jj in range(m):
                    rval += mu[i, j] * mu[ii, jj] * np.abs(D1[i, ii] - D2[j, jj])
    return rval


@njit
def compute_gw_from_transport_sq(mu, D1, D2):
    """
    """
    n = D1.shape[0]
    m = D2.shape[0]
    mu = mu.reshape((n, m))
    rval = 0
    for i in range(n):
        for ii in range(n):
            for j in range(m):
                for jj in range(m):
                    rval += mu[i, j] * mu[ii, jj] * np.abs(D1[i, ii] - D2[j, jj])
    return rval


@njit
def compute_objective(mu, D1, D2):
    """
    """
    n = D1.shape[0]
    m = D2.shape[0]
    rval = np.zeros_like(mu)

    for i in range(n):
        for j in range(m):
            for ii in range(n):
                for jj in range(m):
                    rval[i * m + j] += mu[ii * m + jj] * np.abs(D1[i, ii] - D2[j, jj])
                    #                     rval += mu[i, j] * mu[ii, jj] * np.abs(D1[i, ii] - D2[j, jj])
    return rval


def compute_objective_sq(mu, D1, D2):
    """
    """
    n = D1.shape[0]
    m = D2.shape[0]

    mu = mu.reshape((n, m))
    rval = np.zeros_like(mu)
    A = (D1 ** 2).dot(mu.dot(np.ones(m)))
    B = (D2 ** 2).dot(mu.T.dot(np.ones(n)))
    C = D1.dot((D2).dot(mu.T).T)
    #     for i in range(n):
    #         for j in range(m):
    #             for ii in range(n):
    #                 for jj in range(m):
    #                     rval[i*m + j] += mu[ii*m + jj] * np.abs(D1[i, ii] - D2[j, jj])**2
    #                     rval += mu[i, j] * mu[ii, jj] * np.abs(D1[i, ii] - D2[j, jj])
    return (C + A.reshape(-1, 1) + B.reshape(1, -1)).reshape(-1)


def exact_gw(D1, D2, mu1, mu2, tol=.001, p=1, mode='stable'):
    """ Calculates exact Gromov-Wasserstein distance (to tol) via
    the iterative algorithm discussed in Memoli (2011) and first
    documented in Hendrikson (2016).

    Breaks down the quadratic optimization program min_T T'GT into a series
    of linear programs min_T_(n) T_(n)'GT_(n-1) treating T_(n-1) as fixed on each
    iteration. Initial values of T are populated by computing a lower bound
    which is the transport plan between nodes with the cost being the differences
    between the eccentricities.

    Params
    ---------
    D1/D2 ~ np.array - nxn, mxm
        Distance matrices for each sub-metric space
    mu1/mu2 - np.array - nx1, mx1
        Discrete measure spaces
    tol - float
        Stopping criteria
    p - float
        Not actually implemented - theoretically would parameterize the distance
    mode - string, 'stable', 'fast'
        Determines how to process the enormous G matrix. Fast populates G once
        and stores the enormous matrix - this crashes a lot. Stable calculates the
        values on each iteration without ever instantiating all of G.

    returns:
    ----------
    Gromov-Wasserstein distance (I hope)


    """
    print 'Starting'
    solver = pywraplp.Solver('GW',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = solver.Objective()
    change = 1

    print 'FLP'
    # FLB init
    S1 = (D1 * mu1).sum(axis=1)
    S2 = (D2 * mu2).sum(axis=1)
    obj = np.abs(np.subtract.outer(S1, S2)).reshape(-1)
    params = []
    for i in range(mu1.shape[0]):
        for j in range(mu2.shape[0]):
            var = solver.NumVar(0.0, 1.0, 'mu%s_%s' % (i, j))
            objective.SetCoefficient(var, obj[i * mu2.shape[0] + j])
            params.append(var)
    objective.SetMinimization()
    constraint_list = make_constraints(mu1, mu2, params, solver)  # populate!

    solver.Solve()
    results = []
    for p in params:
        results.append(p.SolutionValue())

    # Solve first iter
    print 'First Real'
    #     G = np.abs(np.kron(D1, np.ones(D2.shape))
    #            - np.kron(np.ones(D1.shape), D2))
    res = []
    #     obj = G.dot(np.array(results))
    obj = compute_objective_sq(np.array(results), D1, D2)
    for i in range(mu1.shape[0]):
        for j in range(mu2.shape[0]):
            objective.SetCoefficient(params[i * mu2.shape[0] + j], obj[i * mu2.shape[0] + j])

    solver.Solve()
    results = []
    result_value = 0
    for p in params:
        results.append(p.SolutionValue())
        result_value += p.SolutionValue() * objective.GetCoefficient(p)
    res.append(result_value)

    print 'Starting iters'
    while tol < change:
        # Reset coeffs
        #         obj = G.dot(np.array(results))
        obj = compute_objective_sq(np.array(results), D1, D2)
        for i in range(mu1.shape[0]):
            for j in range(mu2.shape[0]):
                objective.SetCoefficient(params[i * mu2.shape[0] + j], obj[i * mu2.shape[0] + j])
        # Solve with new coeff
        objective.SetMinimization()
        solver.Solve()
        results = []
        result_value = 0
        for p in params:
            results.append(p.SolutionValue())
            result_value += p.SolutionValue() * objective.GetCoefficient(p)
        res.append(result_value)
        change = res[-2] - res[-1]
        print 'Change', change

    results = np.array(results)
    #     print .5 * check_compute(results, D1, D2, mu1, mu2)
    #     return results.T.dot(G.dot(results))
    return .5 * compute_gw_from_transport(results, D1, D2)


def lb_fused_gw(D1, D2, Dp, mu1, mu2, tol=.001, p=1, mode='stable'):
    """ Calculates exact Gromov-Wasserstein distance (to tol) via
    the iterative algorithm discussed in Memoli (2011) and first
    documented in Hendrikson (2016).

    Breaks down the quadratic optimization program min_T T'GT into a series
    of linear programs min_T_(n) T_(n)'GT_(n-1) treating T_(n-1) as fixed on each
    iteration. Initial values of T are populated by computing a lower bound
    which is the transport plan between nodes with the cost being the differences
    between the eccentricities.

    Params
    ---------
    D1/D2 ~ np.array - nxn, mxm
        Distance matrices for each sub-metric space
    mu1/mu2 - np.array - nx1, mx1
        Discrete measure spaces
    tol - float
        Stopping criteria
    p - float
        Not actually implemented - theoretically would parameterize the distance
    mode - string, 'stable', 'fast'
        Determines how to process the enormous G matrix. Fast populates G once
        and stores the enormous matrix - this crashes a lot. Stable calculates the
        values on each iteration without ever instantiating all of G.

    returns:
    ----------
    Gromov-Wasserstein distance (I hope)


    """
    #     print 'Starting'
    solver = pywraplp.Solver('GW',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = solver.Objective()
    change = 1

    #     print 'FLP'
    # FLB init
    S1 = (D1 * mu1).sum(axis=1)
    S2 = (D2 * mu2).sum(axis=1)
    obj = (np.abs(np.subtract.outer(S1, S2)) + Dp).reshape(-1)
    transport_matrix = np.zeros((mu1.shape[0], mu2.shape[0]))
    params = []
    for i in range(mu1.shape[0]):
        for j in range(mu2.shape[0]):
            var = solver.NumVar(0.0, 1.0, 'mu%s_%s' % (i, j))
            objective.SetCoefficient(var, obj[i * mu2.shape[0] + j])
            params.append(var)
    objective.SetMinimization()
    constraint_list = make_constraints(mu1, mu2, params, solver)  # populate!

    solver.Solve()
    results = []
    result_value = 0
    for i, p in enumerate(params):
        x = i % mu2.shape[0]
        y = i // mu2.shape[0]
        #         print x, y, p
        transport_matrix[y, x] = p.SolutionValue()
        results.append(p.SolutionValue())
        result_value += p.SolutionValue() * objective.GetCoefficient(p)
    # print p.SolutionValue(), objective.GetCoefficient(p)
    return result_value, transport_matrix


def lb_fused_gw_pot(D1, D2, Dp, mu1, mu2, tol=.001, p=1, mode='stable'):
    """ Calculates exact Gromov-Wasserstein distance (to tol) via
    the iterative algorithm discussed in Memoli (2011) and first
    documented in Hendrikson (2016).

    Breaks down the quadratic optimization program min_T T'GT into a series
    of linear programs min_T_(n) T_(n)'GT_(n-1) treating T_(n-1) as fixed on each
    iteration. Initial values of T are populated by computing a lower bound
    which is the transport plan between nodes with the cost being the differences
    between the eccentricities.

    Params
    ---------
    D1/D2 ~ np.array - nxn, mxm
        Distance matrices for each sub-metric space
    mu1/mu2 - np.array - nx1, mx1
        Discrete measure spaces
    tol - float
        Stopping criteria
    p - float
        Not actually implemented - theoretically would parameterize the distance
    mode - string, 'stable', 'fast'
        Determines how to process the enormous G matrix. Fast populates G once
        and stores the enormous matrix - this crashes a lot. Stable calculates the
        values on each iteration without ever instantiating all of G.

    returns:
    ----------
    Gromov-Wasserstein distance (I hope)


    """
    #     print 'Starting'
    #     solver = pywraplp.Solver('GW',
    #                        pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    #     objective = solver.Objective()
    #     change = 1

    #     print 'FLP'
    # FLB init
    S1 = (D1 * mu1).sum(axis=1)
    S2 = (D2 * mu2).sum(axis=1)
    cost = np.abs(np.subtract.outer(S1, S2)) + Dp
    score, log = ot.emd2(mu1, mu2, cost, log=True, return_matrix=True)


def node_merge_emd(net1, net2, metric_space):
    a1 = np.diag(net1.adjacency_matrix.toarray())
    a1 = a1 / a1.sum()
    a2 = np.diag(net2.adjacency_matrix.toarray())
    a2 = a2 / a2.sum()

    a1aug = np.concatenate([a1, np.zeros_like(a2)])
    a2aug = np.concatenate([np.zeros_like(a1), a2])
    dists = network_merge_distance(net1, net2, metric_space)
    a = np.hstack([np.zeros((a1.shape[0], a1.shape[0])), dists])
    b = np.hstack([dists.T, np.zeros((a2.shape[0], a2.shape[0]))])
    c = np.vstack([a, b])
    return pyemd.emd(a1aug, a2aug, c)


def node_merge_truncated_emd(net1, net2, metric_space):
    a1 = np.diag(net1.adjacency_matrix.toarray())
    a1 = a1 / a1.sum()
    a2 = np.diag(net2.adjacency_matrix.toarray())
    a2 = a2 / a2.sum()

    a1aug = np.concatenate([a1, np.zeros_like(a2)])
    a2aug = np.concatenate([np.zeros_like(a1), a2])
    dists = np.minimum(network_merge_distance(net1, net2, metric_space), 2)
    a = np.hstack([np.zeros((a1.shape[0], a1.shape[0])), dists])
    b = np.hstack([dists.T, np.zeros((a2.shape[0], a2.shape[0]))])
    c = np.vstack([a, b])
    return pyemd.emd(a1aug, a2aug, c)


def build_induced_graph(net1, merge_sets, metric_space):
    netx1 = nx.from_scipy_sparse_matrix(net1.adjacency_matrix)
    partition_list = []
    for node_set in merge_sets:
        new_row_set = set(net1.node_row_matrix[list(node_set)].nonzero()[1].tolist())
        new_partition = Partition(subset=new_row_set)
        partition_list.append(new_partition)
    new_network = Network(metric_space, [Cover(partition_list)], NotClusterer(), False)
    return new_network


def component_merge_emd(net1, net2, metric_space):
    netx1 = nx.from_scipy_sparse_matrix(net1.adjacency_matrix)
    netx2 = nx.from_scipy_sparse_matrix(net2.adjacency_matrix)

    cc1 = list(nx.connected_components(netx1))
    cc2 = list(nx.connected_components(netx2))

    net1 = build_induced_graph(net1, cc1, metric_space)
    net2 = build_induced_graph(net2, cc2, metric_space)

    a1 = np.diag(net1.adjacency_matrix.toarray())
    a1 = a1 / a1.sum()
    a2 = np.diag(net2.adjacency_matrix.toarray())
    a2 = a2 / a2.sum()

    a1aug = np.concatenate([a1, np.zeros_like(a2)])
    a2aug = np.concatenate([np.zeros_like(a1), a2])

    dists = network_merge_distance(net1, net2, metric_space)

    a = np.hstack([np.zeros((a1.shape[0], a1.shape[0])), dists])
    b = np.hstack([dists.T, np.zeros((a2.shape[0], a2.shape[0]))])
    c = np.vstack([a, b])
    return pyemd.emd(a1aug, a2aug, c)


def node_merge_ultra_emd(net1, net2, metric_space):
    a1 = np.diag(net1.adjacency_matrix.toarray())
    a1 = a1 / a1.sum()
    a2 = np.diag(net2.adjacency_matrix.toarray())
    a2 = a2 / a2.sum()

    a1aug = np.concatenate([a1, np.zeros_like(a2)])
    a2aug = np.concatenate([np.zeros_like(a1), a2])
    dists = network_merge_distance(net1, net2, metric_space)

    a = np.hstack([np.zeros((a1.shape[0], a1.shape[0])), dists])
    b = np.hstack([dists.T, np.zeros((a2.shape[0], a2.shape[0]))])
    c = np.vstack([a, b])
    return pyemd.emd(a1aug, a2aug, c)


def jacc_emd(net1, net2, metric_space):
    a1 = np.diag(net1.adjacency_matrix.toarray())
    a1 = a1 / a1.sum()
    a2 = np.diag(net2.adjacency_matrix.toarray())
    a2 = a2 / a2.sum()

    a1aug = np.concatenate([a1, np.zeros_like(a2)])
    a2aug = np.concatenate([np.zeros_like(a1), a2])
    dists = 1 - nodewise_jaccard(net1, net2)
    a = np.hstack([np.zeros((a1.shape[0], a1.shape[0])), dists])
    b = np.hstack([dists.T, np.zeros((a2.shape[0], a2.shape[0]))])
    c = np.vstack([a, b])
    return pyemd.emd(a1aug, a2aug, c)


def maximum(A, B):
    BisBigger = A - B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def gromov_wasserstein(net1, net2, metric_space):
    C1 = network_merge_distance(net1, net1, metric_space)
    C1 = C1  # /C1.max()
    C1_span = sparse.csgraph.minimum_spanning_tree(C1)
    aug_adj1 = maximum(net1.adjacency_matrix, C1_span)
    dists1 = np.full_like(metric_space.max())
    dists1[C1_span.nonzero()] = np.max(metric_space)
    dists1[net1.adjacency_matrix.nonzero()] = C1[net1.adjacency_matrix.nonzero()]
    dists1_short = sparse.csgraph.shortest_path(dists1, directed=False)
    dists1_short /= metric_space.max()
    #     print C1
    C2 = network_merge_distance(net2, net2, metric_space)
    C2 = C2  # /C2.max()
    C2_span = sparse.csgraph.minimum_spanning_tree(C2)
    dists2 = np.full_like(metric_space.max())
    dists2[C2_span.nonzero()] = np.max(metric_space)
    dists2[net2.adjacency_matrix.nonzero()] = C2[net2.adjacency_matrix.nonzero()]
    dists2_short = sparse.csgraph.shortest_path(dists2, directed=False)
    dists2_short /= metric_space.max()
    #     print C2

    p = np.diag(net1.adjacency_matrix.toarray())
    p = p / p.sum()

    q = np.diag(net2.adjacency_matrix.toarray())
    q = q / q.sum()
    #     print 'started'
    #     gw_dist = ot.gromov_wasserstein2(dists1_short, dists2_short, p, q,
    #                                      'kl_loss', epsilon=5e-2, verbose = False)
    gw_dist = exact_gw(dists1_short, dists2_short, p, q, tol=.1)
    return gw_dist


def gromov_wasserstein_aug(net1, net2, metric_space):
    C1 = network_merge_distance(net1, net1, metric_space)
    C1 = C1  # /C1.max()
    C1_span = sparse.csgraph.minimum_spanning_tree(C1)
    aug_adj1 = maximum(net1.adjacency_matrix, C1_span)
    dists1 = np.full_like(C1, metric_space.max())
    #     dists1[C1_span.nonzero()] = np.max(metric_space)
    dists1[aug_adj1.nonzero()] = C1[aug_adj1.nonzero()]
    dists1_short = sparse.csgraph.shortest_path(dists1, directed=False)
    #     dists1_short /= metric_space.max()
    #     print C1
    C2 = network_merge_distance(net2, net2, metric_space)
    C2 = C2  # /C2.max()
    C2_span = sparse.csgraph.minimum_spanning_tree(C2)
    aug_adj2 = maximum(net2.adjacency_matrix, C2_span)
    dists2 = np.full_like(C2, metric_space.max())
    #     dists2[C2_span.nonzero()] = np.max(metric_space)
    dists2[aug_adj2.nonzero()] = C2[aug_adj2.nonzero()]
    dists2_short = sparse.csgraph.shortest_path(dists2, directed=False)
    #     dists2_short /= metric_space.max()
    #     print C2

    p = np.diag(net1.adjacency_matrix.toarray())
    p = p / p.sum()

    q = np.diag(net2.adjacency_matrix.toarray())
    q = q / q.sum()
    #     print 'started'
    #     gw_dist = ot.gromov_wasserstein2(dists1_short, dists2_short, p, q,
    #                                      'kl_loss', epsilon=5e-2, verbose = False)
    gw_dist = exact_gw(dists1_short, dists2_short, p, q, tol=.05)
    return gw_dist


def naw_distance(net1, net2, metric_space, p=None, q=None):
    C1 = network_merge_distance(net1, net1, metric_space)
    C1 = C1  # /C1.max()
    C1_span = sparse.csgraph.minimum_spanning_tree(C1)
    aug_adj1 = net1.adjacency_matrix
    dists1 = np.full_like(C1, np.inf)
    #     dists1[C1_span.nonzero()] = np.max(metric_space)
    dists1[aug_adj1.nonzero()] = C1[aug_adj1.nonzero()]
    dists1_short = sparse.csgraph.shortest_path(dists1, directed=False)
    dists1_short[np.isinf(dists1_short)] = 0
    #     dists1_short /= metric_space.max()
    #     print C1
    C2 = network_merge_distance(net2, net2, metric_space)
    C2 = C2  # /C2.max()
    C2_span = sparse.csgraph.minimum_spanning_tree(C2)
    aug_adj2 = net2.adjacency_matrix
    dists2 = np.full_like(C2, np.inf)
    #     dists2[C2_span.nonzero()] = np.max(metric_space)
    dists2[aug_adj2.nonzero()] = C2[aug_adj2.nonzero()]
    dists2_short = sparse.csgraph.shortest_path(dists2, directed=False)
    dists2_short[np.isinf(dists2_short)] = 0

    C3 = network_merge_distance(net1, net2, metric_space)
    if p is None or q is None:
        p = np.diag(net1.adjacency_matrix.toarray())
        p = p / p.sum()

        q = np.diag(net2.adjacency_matrix.toarray())
        q = q / q.sum()

    gw_dist, params = lb_fused_gw(dists1_short, dists2_short, C3, p, q, tol=.05)
    return .5 * gw_dist, params


def pot_wasserstein_aug(net1, net2, metric_space, p=None, q=None):
    C3 = network_merge_distance(net1, net2, metric_space)
    if p is None or q is None:
        p = np.diag(net1.adjacency_matrix.toarray())
        p = p / p.sum()

        q = np.diag(net2.adjacency_matrix.toarray())
        q = q / q.sum()
    # print 'started'
    #     gw_dist = ot.gromov_wasserstein2(dists1_short, dists2_short, p, q,
    #                                      'kl_loss', epsilon=5e-2, verbose = False)
    gw_dist = ot.emd2(p, q, C3)
    params = ot.emd(p, q, C3)
    return gw_dist, params