import argparse
import logging
import math
import os
import sys

import nni
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from script.dataloader import load_adj, load_data, data_transform, data_transform_metro
from script.utility import calc_gso, calc_metric, StandardScaler, MAELoss
from script.visualize import progress_bar
from model.model import STGCN

import os

cpu_num = 8  # 8 thread for 1 GPU
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--enable_nni', type=bool, default=False, help='enable nni experiment')
    parser.add_argument('--dataset', type=str, default='sh-metro', choices=['sh-metro','metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--his', type=int, default=60, help='minute')
    parser.add_argument('--pred', type=int, default=15, help='minute')

    parser.add_argument('--Kt', type=int, default=2, choices=[2, 3], help='kernel size in temporal conv')
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--CTO', type=int, default=128, help='Channels in Temporal conv of Output layer')

    # parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])

    parser.add_argument('--Ks', type=int, default=4,
                        help='kernel size in Spatial conv, Note that the Ks equals to 2 when the graph conv type '
                             'is graph conv (1st approximation cheb graph conv)')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv', 'graph_conv'])

    # parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
    #                     choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])

    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    # parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50, help='epochs, default as 30')
    # parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')

    # parser.add_argument('--step_size', type=int, default=10)
    # parser.add_argument('--gamma', type=float, default=0.95)
    # parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()

    if args.enable_nni:
        RCV_CONFIG = nni.get_next_parameter()
        # RCV_CONFIG = {'batch_size': 16, 'optimizer': 'Adam'}

        parser.set_defaults(**RCV_CONFIG)
        args = parser.parse_args()

    if 'metro' not in args.dataset:
        args.in_channel = args.out_channel = 1
        args.time_intvl = 5
    else:
        args.in_channel = args.out_channel = 2
        args.time_intvl = 15

    args.n_his = int(args.his / args.time_intvl)
    args.n_pred = int(args.pred / args.time_intvl)

    print('Training configs: {}'.format(args))

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # blocks: settings of channel size in network in order
    blocks = [[args.in_channel]]
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])

    # This value is the channel in Temporal conv of output layer
    # and the value seems NOT declare in the paper
    # so makes it a hyperparameter;
    # The channel of the last fc layer should match the size of n_pred
    blocks.append([args.CTO, args.n_pred])

    return args, device, blocks


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


def prepare_model(args, blocks, n_vertex, device, zscore):
    mean, std = zscore.data_info()
    mean, std = mean.to(device), std.to(device)
    loss = MAELoss(mean, std)

    # ckpt_name = 'Kt_' + str(args.Kt) + '_Ks_' + str(args.Ks) + '_ckpt.pth'
    ckpt_name = f'Kt_{args.Kt}_Ks_{args.Ks}_ckpt.pth'
    model = STGCN(args, blocks, n_vertex).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return loss, model, optimizer, ckpt_name


def train(loss, args, optimizer, model, train_iter, val_iter, test_iter, zscore, ckpt_name):
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, channel, n_time, n_vertex]
    """
    best_point = 10e5
    for epoch in range(args.epochs):
        l_sum = 0.0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for batch_idx, (x, y) in enumerate(train_iter):
            y_pred = model(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # l_sum += l.item() * y.shape[0]
            l_sum += l.item()

            rmse, mae, mape = calc_metric(y, y_pred, zscore)

            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(train_iter), 'Train loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))
        print('epoch', epoch + 1)
        val_loss, val_mae = evaluation(model, ckpt_name, 'model_dist', val_iter, zscore, args, type='validation')
        print()
        if args.enable_nni:
            nni.report_intermediate_result(val_mae)

        if val_mae < best_point:
            best_point = val_mae
            model_dist = model.state_dict()

    test_loss, test_mae = evaluation(model, ckpt_name, model_dist, test_iter, zscore, args, type='test')
    if args.enable_nni:
        nni.report_final_result(test_mae)


def evaluation(model, ckpt_name, model_dist, iter, zscore, args, type, saved=True):
    model.eval()
    l_sum, mae_sum = 0.0, 0.0

    if type == 'test':
        # load best model in train
        model.load_state_dict(model_dist)
        print('Test: best train model loaded')

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(iter):
            y_pred = model(x)
            l = loss(y_pred, y)
            # l_sum += l.item() * y.shape[0]
            l_sum += l.item()

            rmse, mae, mape = calc_metric(y, y_pred, zscore)
            mae_sum += mae
            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(iter), str(type) + ' loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))

        print()
        val_mae = mae_sum / len(iter)

        if saved and type == 'test':
            try:
                checkpoint = torch.load('./checkpoint/' + ckpt_name)
                val_loss = checkpoint['loss(mae)']
                print('Found local model dist')
                if val_mae < val_loss:
                    print('Get better model, saving')
                    save_model(args, model, val_mae)
            except:
                save_model(args, model, val_mae)
                print('Local model dist not found, saving...')

        print(str(type) + '_mae', val_mae)
    return l_sum / len(iter), val_mae


def save_model(args, model, val_loss):
    """
    test loss 修改
    """
    checkpoint = {
        'config_args': args,
        'net': model.state_dict(),
        'loss(mae)': val_loss,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(checkpoint, './checkpoint/' + ckpt_name)


if __name__ == '__main__':
    # Logging
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_prepare(args, device)
    loss, model, optimizer, ckpt_name = prepare_model(args, blocks, n_vertex, device, zscore)

    train(loss, args, optimizer, model, train_iter, val_iter, test_iter,zscore, ckpt_name)
    # (loss, args, optimizer, model, train_iter, val_iter, test_iter, zscore, ckpt_name):
    # test(zscore, loss, model, test_iter, args)
