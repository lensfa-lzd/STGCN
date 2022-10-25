import argparse
import logging
import math
import os

import nni
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from script.dataloader import load_adj, load_data, data_transform
from script.utility import calc_gso, calc_metric
from script.visualize import progress_bar
from model.model import STGCN


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--enable_nni', type=bool, default=False, help='enable nni experiment')
    parser.add_argument('--dataset', type=str, default='pemsd7-m', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5, help='means N minutes')
    parser.add_argument('--Kt', type=int, default=3, choices=[2, 3], help='kernel size in temporal conv')
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--CTO', type=int, default=64, help='Channels in Temporal conv of Output layer')

    # parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])

    parser.add_argument('--Ks', type=int, default=3,
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10, help='epochs, default as 30')
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
    blocks = [[1]]
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

    # shape of vel.csv(data) is [num_of_data, num_vertex]
    num_of_data = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]

    '''
        The time period used is from 1st
        July to 31st August, 2014 except the weekends. We select the
        first month of historical speed records as training set, and the
        rest serves as validation and test set respectively.
    '''
    # As described in the paper, the author seems not stating out the radio between validation and test set
    # So we assume the validation:test is 1:1
    train_radio = 2
    val_radio = 1
    test_radio = 1

    radio_sum = train_radio + val_radio + test_radio

    train_radio /= radio_sum
    val_radio /= radio_sum
    test_radio /= radio_sum

    len_val = int(math.floor(num_of_data * val_radio))
    len_test = int(math.floor(num_of_data * test_radio))
    len_train = int(num_of_data - len_val - len_test)

    train, val, test = load_data(args.dataset, len_train, len_val)
    zscore = StandardScaler()

    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = data_transform(train, args.n_his, args.n_pred)
    x_val, y_val = data_transform(val, args.n_his, args.n_pred)
    x_test, y_test = data_transform(test, args.n_his, args.n_pred)

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


def prepare_model(args, blocks, n_vertex, device):
    loss = nn.MSELoss()
    model = STGCN(args, blocks, n_vertex).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return loss, model, optimizer


def train(loss, args, optimizer, model, train_iter, val_iter, zscore):
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, n_pred, n_vertex]
    """
    best_point = 10e5
    for epoch in range(args.epochs):
        l_sum = 0.0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        diff = []
        model.train()
        for batch_idx, (x, y) in enumerate(train_iter):
            y_pred = model(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # l_sum += l.item() * y.shape[0]
            l_sum += l.item()

            mae, mape, rmse = calc_metric(y, y_pred, zscore)

            if batch_idx % 50 == 0:
                progress_bar(batch_idx, len(train_iter), 'Train loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))

        print()
        val_loss, val_mae = evaluation(model, val_iter, zscore, args, type='validation')
        nni.report_intermediate_result(val_mae)

    if args.enable_nni:
        test_loss, test_mae = evaluation(model, val_iter, zscore, args, type='test')
        nni.report_final_result(test_mae)


def evaluation(model, iter, zscore, args, type, saved=True):
    model.eval()
    l_sum, mae_sum = 0.0, 0.0

    if type == 'test':
        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location='cpu')
        model.load_state_dict(checkpoint['net'])

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(iter):
            y_pred = model(x)
            l = loss(y_pred, y)
            # l_sum += l.item() * y.shape[0]
            l_sum += l.item()

            mae, mape, rmse = calc_metric(y, y_pred, zscore)
            mae_sum += mae
            if batch_idx % 50 == 0:
                progress_bar(batch_idx, len(iter), str(type) + ' loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))
        print()
        val_loss_calc = mae_sum / len(iter)
        try:
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            val_loss = checkpoint['val_lost(mae)']
            if val_loss_calc < val_loss:
                save_model(args, model, val_loss_calc)
        except:
            save_model(args, model, val_loss)

    return l_sum / len(iter), val_loss_calc


def save_model(args, model, val_loss):
    checkpoint = {
        'config_args': args,
        'net': model.state_dict(),
        'val_loss(mae)': val_loss,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(checkpoint, './checkpoint/ckpt.pth')


if __name__ == '__main__':
    # Logging
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_prepare(args, device)
    loss, model, optimizer = prepare_model(args, blocks, n_vertex, device)

    train(loss, args, optimizer, model, train_iter, val_iter, zscore)
    # test(zscore, loss, model, test_iter, args)
