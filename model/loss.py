import torch
import torch.nn as nn
from .functions import SIMSE, DiffLoss, MSE, DiffLoss2


class EncoderDecoderLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, config=None):
        super(EncoderDecoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha_weight = 1.0
        self.beta_weight = 1.0


    def forward(self, targets, predictions):
        t_img, t_label = targets[:2]
        ref_code, rec_img = predictions[:2]

        rec_mse = self.alpha_weight * self.mse_loss(rec_img, t_img)

        total_loss = (
            rec_mse + 0.0
        )

        return (
            total_loss,
            {
                'rec_mse': rec_mse,
            },
        )

class StyleDiffLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, config=None):
        super(EncoderDecoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.diff_loss = DiffLoss2()
        self.alpha_weight = 1.0
        self.beta_weight = 1.0


    def forward(self, targets, predictions):
        t_img, t_label = targets[:2]
        ref_code, rec_img, style_embs, text_embs = predictions

        rec_mse = self.alpha_weight * self.mse_loss(rec_img, t_img)
        diff_loss = self.beta_weight * self.diff_loss(style_embs, text_embs)

        total_loss = (
            rec_mse + diff_loss
        )

        return (
            total_loss,
            {
                'rec_mse': rec_mse,
                'diff': diff_loss,
            },
        )
