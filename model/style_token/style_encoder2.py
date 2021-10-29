# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Codes modified from espnet2/tts/gst/style_encoder.py

"""Style encoder of GST-Tacotron."""

from typeguard import check_argument_types
from typing import Sequence, Callable

import torch

from .attention import MultiHeadedAttention
from .attention_gumbel import MultiHeadedAttentionGumbel


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
        # gumbel parameters
        use_gumbel: bool = False,
        gumbel_hard: bool = False,
        gumbel_start_step: int = 55000,
        gumbel_tau_func: Callable = lambda x: 1.0,
        gumbel_activation: str = "softmax",
    ):
        """Initilize style token layer module."""
        assert check_argument_types()
        super(StyleTokenLayer, self).__init__()

        self.gst_heads = gst_heads
        self.gst_tokens = gst_tokens
        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))

        if use_gumbel:
            self.mha = MultiHeadedAttentionGumbel(
                q_dim=ref_embed_dim,
                k_dim=gst_token_dim // gst_heads,
                v_dim=gst_token_dim // gst_heads,
                n_head=gst_heads,
                n_feat=gst_token_dim,
                dropout_rate=dropout_rate,
                # gumbel parameters
                gumbel_hard = gumbel_hard,
                gumbel_start_step = gumbel_start_step,
                gumbel_tau_func = gumbel_tau_func,
                gumbel_activation = gumbel_activation,
            )
        else:
            assert gumbel_activation == 'softmax'  # MultiHeadedAttention does not support sigmoid
            self.mha = MultiHeadedAttention(
                q_dim=ref_embed_dim,
                k_dim=gst_token_dim // gst_heads,
                v_dim=gst_token_dim // gst_heads,
                n_head=gst_heads,
                n_feat=gst_token_dim,
                dropout_rate=dropout_rate,
            )


    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).
            Tensor: Style token weights (B, gst_heads, gst_token_dim).
                Sum of any head's weights is 1.
            Tensor: Style token scores before softmax (B, gst_heads, gst_token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs, weights, scores = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1), weights.squeeze(2), scores.squeeze(2)

    def inference(self, condition: torch.Tensor, method: str='attention') -> torch.Tensor:
        """Calculate weighted style embedding.

        Args:
            condition (Tensor): Reference embeddings (B, gst_heads, gst_tokens).
            method (str): in ['direct','attention']

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        # gst_embs : (gst_tokens, gst_token_dim // gst_heads)  eg. (10, 64)
        # condition : (B, gst_tokens) eg, (32, 10)
        if method == 'direct':
            # NOTE(hellolzc) This method is from open source code,
            # I think the following method is more reasonable
            return torch.matmul(condition, torch.tanh(self.gst_embs))

        assert method == 'attention'
        batch_size = condition.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(hellolzc): apply Tanh as same as the forward function
        weights = condition.view(batch_size, self.gst_heads, 1, self.gst_tokens)  # (#batch, n_head, time1, token_num).
        style_embs = self.mha.inference(gst_embs, weights)

        return style_embs.squeeze(1)
