import math

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn


def estimate_ranks(layer,decompose_rate):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    print(weights.shape)
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    print('unfold_0.shape :',unfold_0.shape[0])
    print('unfold_1.shape :',unfold_1.shape)
    d1 = decompose_rate*unfold_0.shape[0]
    d2 = decompose_rate*unfold_1.shape[0]

    rank_1=math.ceil(d1) #stand for R3
    rank_2=math.ceil(d2) #stand for R4
    if rank_1%2 != 0:
        rank_1 = rank_1 + 1
    if rank_2 %2 != 0:
        rank_2 = rank_2 + 1
    ranks = [rank_1,rank_2]
    return ranks

def estimate_linear_ranks(layer,decompose_rate):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    weights = layer.weight.data
    print(weights.shape)
    unfold_1 = tl.base.unfold(weights, 1)
    print('unfold_1.shape :',unfold_1.shape)
    d2 = decompose_rate*unfold_1.shape[0]
    rank_2=math.ceil(d2) #stand for R4

    if rank_2 %2 != 0:
        rank_2 = rank_2 + 1
    ranks = [rank_2]
    return ranks

def tucker_decomposition_conv_layer(layer,decompose_rate):
    # Gets a conv layer,
    # returns a nn.Sequential object with the Tucker decomposition.

    ranks = estimate_ranks(layer,decompose_rate)

    print(layer, "Auto Estimated ranks", ranks)
    core, [last, first] = partial_tucker(layer.weight.data,modes=[0, 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def tucker_decomposition_linear_layer(layer,decompose_rate):
    ranks = estimate_linear_ranks(layer, decompose_rate)

    print(layer, "Auto Estimated ranks", ranks)
    core, [last] = partial_tucker(layer.weight.data, modes=[ 1], ranks=ranks, init='svd', tol=1)
    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Linear(core.shape[1], \
                                 core.shape[0],
                                 bias=True)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Linear(last.shape[0], \
                                 last.shape[1], bias=False)

    core_layer.bias.data = layer.bias.data


    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [last_layer,core_layer ]
    return nn.Sequential(*new_layers)

def tucker_for_first_linear_layer(layer,decompose_rate):
    ranks = estimate_ranks(layer, decompose_rate)

    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Linear(first.shape[0], \
                                  first.shape[1],
                                 bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Linear(core.shape[1], \
                                 core.shape[0],
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Linear(last.shape[1], \
                                 last.shape[0],  bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def tucker_for_first_conv_layer(layer,decompose_rate):

    ranks = [math.ceil(decompose_rate*64)]

    print(layer, "Auto Estimated ranks", ranks)
    core, [last] = partial_tucker(layer.weight.data,modes=[ 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3


    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=ranks[0], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=True)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=ranks[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=False)

    core_layer.bias.data = layer.bias.data



    last_layer.weight.data = torch.transpose(last, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [last_layer,core_layer]
    return nn.Sequential(*new_layers)