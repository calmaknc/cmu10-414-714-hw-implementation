import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch
import itertools
import os

import needle as ndl
import needle.nn as nn


def test_transformer_layer(batch_size, seq_len, input_dim, num_head, dim_head, hidden_size, causal, dropout, device):
    np.random.seed(19943)
    x = np.random.randn(
        batch_size, seq_len, input_dim
    ).astype(np.float32)
    ndl_x = ndl.Tensor(x, device=device)

    layer = nn.TransformerLayer(
        input_dim, num_head, dim_head, hidden_size,
        dropout=dropout, causal=causal, device=device)

    # layer1_nd = nn.AttentionLayer(q_features=input_dim,
    #                                num_head=num_head,
    #                                dim_head=dim_head,
    #                                out_features=input_dim,
    #                                dropout=dropout,causal=causal,device=device,dtype='float32')

    # layer2_nd = nn.Sequential(
    #             nn.LayerNorm1d(dim=input_dim,device=device,dtype='float32'),
    #             nn.Linear(input_dim,hidden_size,device=device,dtype='float32'),
    #             nn.ReLU(),
    #             nn.Dropout(dropout),
    #             nn.Linear(hidden_size,input_dim,device=device,dtype='float32'),
    #             nn.Dropout(dropout)
    #     )


    # result = layer1_nd(ndl_x) + ndl_x
    # result_reshape = result.reshape((batch_size*seq_len,input_dim))
    
    # result = layer2_nd(result_reshape).reshape((batch_size, seq_len, input_dim)) + result

    result = result.numpy()

    current_input_id = "-".join([str(x) for x in (
        batch_size, seq_len, input_dim, num_head, dim_head, hidden_size, causal, dropout, device
    )])

    labels_path = (
        "./tests/hw4_extra/data/" + 
        "test_transformer_layer-{}.npy"
        .format(current_input_id))

    with open(labels_path, 'rb') as f:
        label_result = np.load(f)

    np.testing.assert_allclose(result, label_result, atol=1e-5, rtol=1e-5)