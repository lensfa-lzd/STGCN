import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class TemporalConvLayer(nn.Module):
    '''
        TemporalConvLayer(Kt, last_block_channel, channels_out, n_vertex)
    '''

    # Temporal Convolution Layer
    #
    #    |------------------------| * Residual Connection *
    #    |                        |
    #    |                  | --- + ---------|
    # ---|---CasualConv2d---|                ⊙ ------>
    #                       | --- Sigmoid ---|

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex

        # use padding to make the n_time unchanged
        minus_after_conv = Kt - 1
        self.padding = minus_after_conv

        # Residual Connection may conduct by 1D conv to match the channel size
        if c_in != c_out:
            self.res_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))
        else:
            self.res_conv = nn.Identity()

        # Should be declare separately, so they will have same parameters
        # Kernel size in nn.Conv2d is declare as [row, column],
        # and setting it as (Kt, 1) meaning conduct 1D conv along the temporal dimension
        # while the size of input is [bs, c_in, ts, n_vertex]
        self.causal_conv = nn.Conv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # use padding to make the n_time unchanged
        # [bs, c_in, n_time, n_vertex]
        # padding (left, right, top, bottom) for [n_time, n_vertex]
        x_pad = F.pad(x, (0, 0, self.padding, 0), mode='constant', value=0)

        x_conv = self.causal_conv(x_pad)
        x_res = self.res_conv(x)

        x_p = x_conv[:, : self.c_out, :, :]
        x_q = x_conv[:, -self.c_out:, :, :]

        # GLU was first purposed in
        # *Language Modeling with Gated Convolutional Networks*.
        # URL: https://arxiv.org/abs/1612.08083
        # Input tensor X is split by a certain dimension into tensor X_a and X_b.
        # In PyTorch, GLU is defined as X_a ⊙ Sigmoid(X_b).
        # URL: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
        # (x_p + x_in) ⊙ Sigmoid(x_q)

        # print(3, x_p.shape, x_res.shape, x_q.shape)
        x = torch.mul((x_p + x_res), self.sigmoid(x_q))

        return x


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.initialize_parameters_parameters()

    def initialize_parameters_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # [batch_size, n_time, n_vertex, channel]

        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)

        return cheb_graph_conv


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.initialize_parameters()

    def initialize_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # [batch_size, n_time, n_vertex, channel]
        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphConvLayer(nn.Module):
    # the size of input is [batch_size, channel, n_time, n_vertex] which remain the same with the dataset
    # So its necessary to conduct a transpose for input before and after the Graph conv

    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        if self.graph_conv_type == 'cheb_graph_conv':
            self.cheb_graph_conv = ChebGraphConv(c_in, c_out, Ks, gso)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_in, c_out, gso)

    def forward(self, x):
        # size of input and output is [batch_size, channel, n_time, n_vertex]
        # but size for graph conv is [batch_size, n_time, n_vertex, channel]
        # So rearrange the input here
        x = torch.einsum('bctv->btvc', x)

        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x)

        # arrange back to [batch_size, channel, n_time, n_vertex]
        x_gc = torch.einsum('btvc->bctv', x_gc)
        return x_gc


class STConvBlock(nn.Module):
    # Overall pipeline
    # T: Gated Temporal Convolution Layer
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer
    # N: Layer Normolization
    '''
        Kt: kernel size of temporal conv
        Ks: kernel size of spatial conv
        channels: channels in layers e.g., [64, 16, 64]
    '''

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, graph_conv_type, gso):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex)

        # [n_vertex, channels[2]] should match the size of the last 2 dimension of input
        # and the shape of x is [batch_size, channel, n_time, n_vertex]
        # so a reshape for x is needed before the operation
        # Note that the result is different when norm along the spatial axis or the temporal axis
        self.layer_norm = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)

        # [batch_size, channel, n_time, n_vertex] -> [batch_size, n_time, n_vertex, channel] -> origin
        x = torch.einsum('bctv->btvc', x)
        x = self.layer_norm(x)
        x = torch.einsum('btvc->bctv', x)

        return x


class OutputBlock(nn.Module):
    # Overall pipeline
    # T: Gated Temporal Convolution Layer
    # F: Fully-Connected Layer

    '''
        channels: channels in layers e.g., [args.CTO, args.n_pred]
        CTO: Channel of Temporal conv in the Output layer
    '''

    def __init__(self, Kt, last_block_channel, channels, n_vertex, n_time):
        super(OutputBlock, self).__init__()
        self.tmp_conv = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex)
        self.fc = nn.Linear(in_features=channels[0]*n_time, out_features=channels[1])

    def forward(self, x):
        x = self.tmp_conv(x)

        batch_size = x.shape[0]
        n_vertex = x.shape[3]

        # [batch_size, channel, n_time, n_vertex] -> [batch_size, n_vertex, n_time, channel]
        # -> [batch_size, n_vertex, n_time*channel] -> [batch_size, n_vertex, n_pred]
        # -> [batch_size, n_pred, n_vertex]
        x = torch.einsum('bctv->bvtc', x)
        x = x.reshape([batch_size, n_vertex, -1])
        x = self.fc(x)
        x = torch.einsum('bvt->btv', x)

        return x
