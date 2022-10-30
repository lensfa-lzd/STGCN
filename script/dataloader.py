import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228

    id = sp.identity(n_vertex, format='csc')
    # The "original" adjacency matrix (adj) should NOT be have a self-loop/self-connection (value 1 fill in diagonal)
    # However all adjs in dataset above have a self-loop/self-connection
    # So removing the self-connection here makes it equal to the adjacency matrix (W) in the paper
    adj = adj - id

    # adj is weighted in dataset above
    return adj, n_vertex


def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    # shape of vel [num_of_data, num_vertex]
    vel = torch.load(os.path.join(dataset_path, 'vel.pth'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
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