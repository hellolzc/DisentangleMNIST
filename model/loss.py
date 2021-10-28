import torch
import torch.nn as nn
from functions import SIMSE, DiffLoss, MSE


class EncoderDecoderLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, config=None):
        super(EncoderDecoderLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha_weight = 1.0
        self.beta_weight = 1.0


    def forward(self, targets, predictions):
        t_img, t_label = targets
        ref_code, rec_img = predictions

        rec_mse = self.alpha_weight * self.mse_loss(rec_img, t_img)
        
        total_loss = (
            rec_mse + 0.0
        )

        return (
            total_loss,
            rec_mse,
        )