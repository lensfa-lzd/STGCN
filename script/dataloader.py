import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from einops import repeat

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = torch.load(os.path.join(dataset_path, 'adj.pth'))
    adj = adj.numpy()

    if dataset_name == 'pems07':
        n_vertex = 883
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'hz-metro':
        n_vertex = 80
    elif dataset_name == 'sh-metro':
        n_vertex = 288

    id = np.eye(n_vertex)

    # The "original" adjacency matrix (adj) should NOT have a self-loop/self-connection (value 1 fill in diagonal)
    # However all adjs in dataset above have a self-loop/self-connection
    # So removing the self-connection here makes it equal to the adjacency matrix (W) in the paper
    adj = adj - id

    adj[adj > 0] = 1  # 0/1 matrix, symmetric

    # adj is weighted in dataset above
    return adj, n_vertex


def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    # shape of vel [num_of_data, num_vertex, channel]
    vel = torch.load(os.path.join(dataset_path, 'vel.pth'))
    n_vertex = vel.shape[-2]

    # shape of time_index [num_of_data, 2]
    # time_index [..., 0] for dayofweek
    # time_index [..., 1] for timeofday
    if 'metro' not in dataset_name:
        time_index = pd.read_hdf(os.path.join(dataset_path, 'time_index.h5'))
        time_index = calc_te(time_index)

        # shape of te [num_of_data, num_vertex, 2]
        # te [..., 0] for dayofweek
        # te [..., 1] for timeofday
        te = repeat(time_index, 't c -> t v c', v=n_vertex)
    else:
        te = torch.load(os.path.join(dataset_path, 'te.pth'))
        te = repeat(te, 'd t c -> d t v c', v=n_vertex)

    train = (vel[: len_train], te[: len_train])
    val = (vel[len_train: len_train + len_val], te[len_train: len_train + len_val])
    test = (vel[len_train + len_val:], te[len_train + len_val:])
    return train, val, test


def data_transform(data, n_his, n_pred):
    # produce data slices for x_data and y_data

    # shape of data [num_of_data, num_vertex, channel]
    channel = data.shape[-1]
    n_vertex = data.shape[1]
    len_record = data.shape[0]
    num = len_record - n_his - n_pred

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, n_pred, n_vertex]
    x = torch.zeros([num, n_his, n_vertex, channel])
    y = torch.zeros([num, n_pred, n_vertex, channel])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail]
        y[i, :, :, :] = data[tail: tail + n_pred]

    x = torch.einsum('btvc->bctv', x).float()
    y = torch.einsum('btvc->bctv', y).float()

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    return x, y


def data_transform_metro(data, n_his, n_pred):
    # produce data slices for x_data and y_data

    # shape of data [day, num_of_data, num_vertex, channel]
    num_day = data.shape[0]
    channel = data.shape[-1]
    n_vertex = data.shape[2]
    len_record = data.shape[1]
    num = len_record - n_his - n_pred

    x_list = []
    y_list = []

    for day in range(num_day):
        x = torch.zeros([num, n_his, n_vertex, channel])
        y = torch.zeros([num, n_pred, n_vertex, channel])

        for i in range(num):
            head = i
            tail = i + n_his
            x[i, :, :, :] = data[day, head: tail, :, :]
            y[i, :, :, :] = data[day, tail: tail + n_pred, :, :]

        x_list.append(x)
        y_list.append(y)

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    x = torch.einsum('btvc->bctv', x).float()
    y = torch.einsum('btvc->bctv', y).float()

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    return x, y