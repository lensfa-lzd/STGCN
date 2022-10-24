from torch import nn

from model import layers


class STGCN(nn.Module):
    # Using Chebyshev Polynomials Approximation (ChebGraphConv)
    # or 1st-order Approximation (GraphConv) for GCN

    # Overall pipeline
    # T: Gated Temporal Convolution Layer
    # G: Graph Convolution Layer
    # T: Gated Temporal Convolution Layer
    # N: Layer Normolization

    # T: Gated Temporal Convolution Layer
    # G: Graph Convolution Layer
    # T: Gated Temporal Convolution Layer
    # N: Layer Normolization

    # T: Gated Temporal Convolution Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCN, self).__init__()
        modules = []

        # block ([1], [64, 16, 64], [64, 16, 64], [args.CTO, args.n_pred])
        # 2 STConvBlock in Network
        for l in range(2):
            # STConvBlock(Kt, Ks, n_vertex, last_block_channel, channels, graph_conv_type, gso)
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l + 1],
                                              args.graph_conv_type, args.gso))
        self.st_blocks = nn.Sequential(*modules)

        # OutputBlock(Kt, last_block_channel, channels, n_vertex, n_time_final):
        self.output_layer = layers.OutputBlock(args.Kt, blocks[2][-1], blocks[3], n_vertex, args.n_his)

    def forward(self, x):
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        # size of y/target [batch_size, n_pred, n_vertex]
        x = self.st_blocks(x)
        x = self.output_layer(x)

        return x
