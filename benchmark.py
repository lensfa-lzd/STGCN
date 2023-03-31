import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from model.model import STGCN
from script.dataloader import load_adj, load_data, data_transform, data_transform_metro
from script.utility import calc_gso, calc_metric, StandardScaler
from script.visualize import progress_bar

cpu_num = 8  # 8 thread for 1 GPU
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)


def get_parameters(setting):
    checkpoint = torch.load('checkpoint/' + setting.model_name, map_location=device)
    loss = checkpoint['loss(mae)']

    args = checkpoint['config_args']
    args.enable_nni = False

    # mask = args.mask
    # args.mask = None
    blocks = [[args.in_channel]]
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])
    blocks.append([args.CTO, args.n_pred])

    print('Loaded configs: {}'.format(args))
    print('load loss', loss)
    print()

    return args, checkpoint, blocks


def data_prepare(args, device):
    adj, n_vertex = load_adj(args.dataset)
    # parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
    #                     choices=['cheb_graph_conv', 'graph_conv'])
    if args.graph_conv_type == 'cheb_graph_conv':
        gso_type = 'sym_norm_lap'
    else:
        # graph_conv_type == graph_conv (1st approximation cheb graph conv)
        gso_type = 'sym_renorm_adj'

    gso = calc_gso(adj, gso_type)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)

    num_of_data = torch.load(os.path.join(dataset_path, 'vel.pth')).shape[0]

    if 'metro' not in args.dataset:
        train_radio = 7
        val_radio = 1
        test_radio = 2
    elif 'hz' in args.dataset:
        # for a full day
        train_radio = 18
        val_radio = 2
        test_radio = 5
    elif 'sh' in args.dataset:
        # for a full day
        train_radio = 62
        val_radio = 9
        test_radio = 21
    else:
        print('dataset error')
        sys.exit()

    radio_sum = train_radio + val_radio + test_radio

    train_radio /= radio_sum
    val_radio /= radio_sum
    test_radio /= radio_sum

    len_val = int(math.floor(num_of_data * val_radio))
    len_test = int(math.floor(num_of_data * test_radio))
    len_train = int(num_of_data - len_val - len_test)

    train_tuple, val_tuple, test_tuple = load_data(args.dataset, len_train, len_val)

    train, val, test = train_tuple[0], val_tuple[0], test_tuple[0]
    train_te, val_te, test_te = train_tuple[1], val_tuple[1], test_tuple[1]
    # train, val, test = train.to(device), val.to(device), test.to(device)
    # train_te, val_te, test_te = train_te.to(device), val_te.to(device), test_te.to(device)

    # print(train.shape)
    # sys.exit()

    zscore = StandardScaler(train)

    # shape of train/val/test [num_of_data, num_vertex, channel]
    train = zscore.transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    train = torch.cat((train, train_te), dim=-1).float()
    val = torch.cat((val, val_te), dim=-1).float()
    test = torch.cat((test, test_te), dim=-1).float()

    if 'metro' not in args.dataset:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        # size of y/target [batch_size, channel, n_time, n_vertex]
        x_train, y_train = data_transform(train, args.n_his, args.n_pred)
        x_val, y_val = data_transform(val, args.n_his, args.n_pred)
        x_test, y_test = data_transform(test, args.n_his, args.n_pred)
    else:
        x_train, y_train = data_transform_metro(train, args.n_his, args.n_pred)
        x_val, y_val = data_transform_metro(val, args.n_his, args.n_pred)
        x_test, y_test = data_transform_metro(test, args.n_his, args.n_pred)

    y_train, y_val, y_test = y_train[:, :args.in_channel, :, :], y_val[:, :args.in_channel, :, :], y_test[:, :args.in_channel, :, :]

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_iter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_iter = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_iter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    # size of x/input is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, n_pred, n_vertex]
    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex, checkpoint, device):
    model = STGCN(args, blocks, n_vertex).to(device)
    model.load_state_dict(checkpoint['net'])

    return model


def evaluation(model, iter, zscore):
    target, output = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(iter):
            y_pred = model(x)

            progress_bar(batch_idx, len(iter), str(type) + ' loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                         % (0 / (batch_idx + 1), 0, 0, 0))

            target.append(y)
            output.append(y_pred)

    target, output = torch.cat(target, dim=0), torch.cat(output, dim=0)
    rmse, mae, mape = calc_metric(target, output, zscore)
    return mae, mape, rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waiting for a project name')
    parser.add_argument('--model_name', type=str, default='Kt_2_Ks_3_60min_ckpt.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    setting = parser.parse_args()

    device = torch.device(setting.device)
    args, checkpoint, blocks = get_parameters(setting)
    args.batch_size = 4
    n_vertex, zscore, train_iter, val_iter, test_iter = data_prepare(args, device)

    model = prepare_model(args, blocks, n_vertex, checkpoint, device)

    # print('*'*20)
    # print('mae', ' ' * 2, 'mape', ' ' * 1, 'rmse')
    # mae, mape, rmse = evaluation(model, train_iter, zscore, args, se, 'train')
    # print('train')
    # print(format(mae, '.3f'), format(mape, '.3f'), format(rmse, '.3f'))
    #
    # print('*' * 20)
    # mae, mape, rmse = evaluation(model, val_iter, zscore, args, se, 'validation')
    # print('validation')
    # print(format(mae, '.3f'), format(mape, '.3f'), format(rmse, '.3f'))

    print('*' * 20)
    mae, mape, rmse = evaluation(model, test_iter, zscore)
    print('test')
    print(format(mae, '.3f'), format(mape, '.3f'), format(rmse, '.3f'))
