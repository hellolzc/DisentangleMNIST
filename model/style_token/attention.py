#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Codes from espnet/nets/pytorch_backend/transformer/attention.py
# and espnet2/tts/gst/style_encoder.py

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi head attention module with different input dimension.
    
    Args:
        q_dim (int): Dimension of Q.
        k_dim (int): Dimension of K.
        v_dim (int): Dimension of V.
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(q_dim, n_feat)
        self.linear_k = nn.Linear(k_dim, n_feat)
        self.linear_v = nn.Linear(v_dim, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v


    def forward_attention(self, value, scores, mask, weights=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
            weights (torch.Tensor, optional): Attention score after softmax (#batch, n_head, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        if weights is not None:
            assert self.attn.shape == weights.shape
            self.attn = weights

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x), self.attn  # (batch, time1, d_model), (#batch, time1, time2)

    def forward(self, query, key, value, mask, weights=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            weights (torch.Tensor, optional): if is not None: use them to
                compute context vector. (#batch, n_head, time1, time2)

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Attention weights (#batch, n_head, time1, time2).
            torch.Tensor: Attention scores (#batch, n_head, time1, time2).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        output, attn = self.forward_attention(v, scores, mask)
        return output, attn, scores


    def inference(self, value, weights):
        """Transform query, key and value.

        Args:
            value (torch.Tensor): Value tensor (#batch, time2, size).
                As for GST, shape is   (#batch, token_num, token_size).
            weights (torch.Tensor): use them to compute context vector.
                i.e. Attention score after softmax (#batch, n_head, time1, time2).
                As for GST, shape is (#batch, n_head, time1, token_num).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        # forward_qkv -- value
        n_batch = value.size(0)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # forward_attention
        self.attn = weights  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)
