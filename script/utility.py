import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from einops import rearrange


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
    y, y_pred = y.detach().cpu(), y_pred.detach().cpu()

    y = zscore.inverse_transform(y)
    y_pred = zscore.inverse_transform(y_pred)

    metric = Metrics(y, y_pred)
    return metric.all()


def evaluate_metric(y, y_pred, zscore):
    """
        size of y/y_pred [batch_size, n_pred, n_vertex]
    """
    y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()
    y = zscore.inverse_transform(y).reshape(-1)
    y_pred = zscore.inverse_transform(y_pred).reshape(-1)

    # diff =


class MAELoss(torch.nn.Module):
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, channel, n_time, n_vertex]
        Calc MAE by channel
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x, y):
        # print(x.shape, y.shape)
        # print(self.mean, self.std)
        # metro torch.Size([8, 2, 4, 80]) torch.Size([8, 2, 4, 80]) mean [2] std [2]
        # road torch.Size([8, 1, 12, 325]) torch.Size([8, 1, 12, 325]) mean [1] std [1]
        x, y = torch.einsum('bctv->btvc', x), torch.einsum('bctv->btvc', y)
        x = x * self.std + self.mean
        y = y * self.std + self.mean
        mae = torch.absolute(x-y)
        return torch.mean(mae)

    @staticmethod
    def calc_mae(x, y):
        """
        size of x/input is [batch_size, n_time, n_vertex]
        size of y/output/target [batch_size, n_time, n_vertex]
        """
        mae = torch.absolute(x-y)
        return torch.mean(mae)


class StandardScaler:
    """
        input/output shape [num_of_data, num_vertex, channel]
        or [batch_size, channel, n_time, n_vertex]
    """
    def __init__(self, fit_data):
        if len(fit_data.shape) != 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            fit_data = rearrange(fit_data, 'd t v c -> (d t) v c')

        fit_data = rearrange(fit_data, 't v c -> (t v) c')
        self.mean = torch.mean(fit_data, dim=0)
        self.std = torch.std(fit_data, dim=0)

    def transform(self, x):
        if len(x.shape) == 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = (x-self.mean)/self.std
            return rearrange(x, '(t v) c -> t v c', v=v).float()
        else:
            # for metro data [day, n_time, n_vertex, channel]
            v = x.shape[-2]
            batch_size = x.shape[0]
            x = rearrange(x, 'd t v c -> (d t v) c')
            x = (x - self.mean) / self.std
            return rearrange(x, '(d t v) c -> d t v c', d=batch_size, v=v).float()

    def inverse_transform(self, x):
        if len(x.shape) == 3:
            # dataset data arrange by [time, vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = x*self.std + self.mean
            return rearrange(x, '(t v) c -> t v c', v=v)
        else:
            # network output data/target data arrange by [batch_size, channel, time, vertex]
            v = x.shape[-1]
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t v -> (b t v) c')
            x = x * self.std + self.mean
            return rearrange(x, '(b t v) c -> b c t v', b=batch_size, v=v)

    def data_info(self):
        return self.mean, self.std


class Metrics(object):
    """
        masked version error functions partly base on PVCGN
        https://github.com/HCPLab-SYSU/PVCGN

        size of output is [batch_size, channel, n_time, n_vertex]
        size of target [batch_size, channel, n_pred, n_vertex]
        thus axis 1 is "channel"
    """
    def __init__(self, target, output):
        self.target = target
        self.output = output

        # zero value might be slightly change due to Z-score norm
        self.mask = target < 10e-5

    def mse(self):
        mse = torch.square(self.target - self.output)
        mse[self.mask] = 0
        return torch.mean(mse)

    def rmse(self):
        return torch.sqrt(self.mse())

    def mae(self):
        mae = torch.absolute(self.target - self.output)
        mae[self.mask] = 0
        return torch.mean(mae)

    def mape(self):
        mape = torch.absolute((self.target - self.output)/self.target)
        mape[self.mask] = 0
        return torch.mean(mape * 100)

    def all(self):
        rmse = self.rmse().item()
        mae = self.mae().item()
        mape = self.mape().item()

        return rmse, mae, mape