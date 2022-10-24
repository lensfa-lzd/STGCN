import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch


def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    # makes difference when adj is directed, used to average the different value between (i->j) and (j->i)
    # but in this case (data in the data folder) all the adjs are undirected, so you can take adj = dir_adj
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    if gso_type == 'sym_renorm_adj':
        # Adding self-connection
        adj = adj + id

    row_sum = adj.sum(axis=1).A1
    row_sum_inv_sqrt = np.power(row_sum, -0.5)
    row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
    deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')

    # A_{sym} = D^{-0.5} * A * D^{-0.5}
    sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

    if gso_type == 'sym_norm_lap':
        sym_norm_lap = id - sym_norm_adj
        gso = calc_chebynet_gso(sym_norm_lap)
    else:
        gso = sym_norm_adj

    return gso


def calc_chebynet_gso(gso):
    # rescale operation in Chebyshev Polynomials Approximation
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    eigval_max = max(eigsh(A=gso, k=6, which='LM', return_eigenvectors=False))

    gso = 2 * gso / eigval_max - id

    return gso


def calc_metric(y, y_pred, zscore):
    """
        size of y/y_pred [batch_size, n_pred, n_vertex]
    """
    y, y_pred = y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

    y = zscore.inverse_transform(y).reshape(-1)
    y_pred = zscore.inverse_transform(y_pred).reshape(-1)

    diff = np.abs(y - y_pred)
    mae = np.mean(diff)
    mape = np.mean(diff / y) * 100
    mse = diff * diff
    rmse = np.sqrt(np.mean(mse))

    return mae, mape, rmse


def evaluate_metric(y, y_pred, zscore):
    """
        size of y/y_pred [batch_size, n_pred, n_vertex]
    """
    y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()
    y = zscore.inverse_transform(y).reshape(-1)
    y_pred = zscore.inverse_transform(y_pred).reshape(-1)

    # diff =



