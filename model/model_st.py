from typing import Sequence, Callable
import torch
import torch.nn as nn
from .model_ed import Encoder, Decoder, RefEncoder
from .style_token.style_encoder2 import StyleTokenLayer



class StyleEncoder(torch.nn.Module):
    """Style encoder.

    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.

        ref_embed_dim (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
        self,
        gst_tokens: int = 10,
        gst_token_dim: int = 128,
        gst_heads: int = 4,
        ref_embed_dim: int = 128,
        # gumbel parameters
        use_gumbel: bool = False,
        gumbel_hard: bool = False,
        gumbel_start_step: int = 55000,
        gumbel_tau_func: Callable = lambda x: 1.0,
        gumbel_activation: str = "softmax",
    ):

        super(StyleEncoder, self).__init__()

        self.ref_enc = RefEncoder(code_size=ref_embed_dim)
        self.stl = StyleTokenLayer(
            ref_embed_dim=ref_embed_dim,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
            # gumbel parameters
            use_gumbel = use_gumbel,
            gumbel_hard = gumbel_hard,
            gumbel_start_step = gumbel_start_step,
            gumbel_tau_func = gumbel_tau_func,
            gumbel_activation = gumbel_activation,
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            input_data (Tensor): Batch of images (B, channel, image_size, image_size). eg. (64, 3, 28, 28)

        Returns:
            Tensor: Style token embeddings (B, token_dim).
            Tensor: Style token weights (B, n_head, token_dim)
            Tensor: Style token scores (B, n_head, token_dim)

        """
        ref_embs = self.ref_enc(input_data)
        style_embs, weights, scores = self.stl(ref_embs)

        return style_embs, weights, scores


    def inference(self, condition: torch.Tensor, method: str='attention') -> torch.Tensor:
        """Calculate weighted style embedding.

        Args:
            condition (Tensor): Reference embeddings (B, gst_heads, gst_tokens).
            method (str): in ['direct','attention']

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        style_embs = self.stl.inference(condition, method=method)

        return style_embs



class ModelST(nn.Module):
    """Number to image with style token"""
    def __init__(self, config):
        super(ModelST, self).__init__()
        code_size= config['code_size']   # 100, 
        n_class  = config['n_class']     # 10
        token_num = config['token_num']  # 10

        self.encoder = Encoder(n_class, 64, code_size)
        self.decoder = Decoder(code_size)

        self.style_encoder = StyleEncoder(
            gst_token_dim=code_size,
            ref_embed_dim=code_size,
            gst_tokens=token_num,
            gst_heads=1,
        )

    def forward(self, input_data='unused', number=None):
        # encoder
        emb_code = self.encoder(number)
        style_embs, weights, scores = self.style_encoder(input_data)
        # decoder
        union_code = emb_code + style_embs
        rec_img = self.decoder(union_code)

        return emb_code, rec_img



class ModelSVC(nn.Module):
    """Number to image with style variation (Categorical)"""
    def __init__(self, config):
        super(ModelSVC, self).__init__()
        code_size= config['code_size']   # 100, 
        n_class  = config['n_class']     # 10
        token_num = config['token_num']  # 10

        self.encoder = Encoder(n_class, 64, code_size)
        self.decoder = Decoder(code_size)

        self.style_encoder = StyleEncoder(
            gst_token_dim=code_size,
            ref_embed_dim=code_size,
            gst_tokens=token_num,
            gst_heads=1,
            use_gumbel=True,
            gumbel_activation='softmax',
        )

    def forward(self, input_data='unused', number=None):
        # encoder
        emb_code = self.encoder(number)
        style_embs, weights, scores = self.style_encoder(input_data)
        # decoder
        union_code = emb_code + style_embs
        rec_img = self.decoder(union_code)

        return emb_code, rec_img


class ModelSVB(nn.Module):
    """Number to image with style variation (Bernoulli)"""
    def __init__(self, config):
        super(ModelSVB, self).__init__()
        code_size= config['code_size']   # 100, 
        n_class  = config['n_class']     # 10
        token_num = config['token_num']  # 10

        self.encoder = Encoder(n_class, 64, code_size)
        self.decoder = Decoder(code_size)

        self.style_encoder = StyleEncoder(
            gst_token_dim=code_size,
            ref_embed_dim=code_size,
            gst_tokens=token_num,
            gst_heads=1,
            use_gumbel=True,
            gumbel_activation='sigmoid',
        )

    def forward(self, input_data='unused', number=None):
        # encoder
        emb_code = self.encoder(number)
        style_embs, weights, scores = self.style_encoder(input_data)
        # decoder
        union_code = emb_code + style_embs
        rec_img = self.decoder(union_code)

        return emb_code, rec_img

