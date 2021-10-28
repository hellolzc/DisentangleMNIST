import random
import torch
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter


class ExpLogger(SummaryWriter):
    def __init__(self, logdir):
        super(ExpLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate,
                     iteration, scalar_dict=None):
            self.add_scalar("Training/training.loss", reduced_loss, iteration)
            self.add_scalar("Training/grad.norm", grad_norm, iteration)
            self.add_scalar("Training/learning.rate", learning_rate, iteration)

            if scalar_dict is not None:
                for key in scalar_dict:
                    self.add_scalar("Training/training."+key, scalar_dict[key], iteration)

    def log_validation(self, reduced_loss, model, iteration, scalar_dict=None):
        self.add_scalar("Validation/validation.loss", reduced_loss, iteration)

        if scalar_dict is not None:
            for key in scalar_dict:
                self.add_scalar("Validation/validation."+key, scalar_dict[key], iteration)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            if len(value.shape) < 2:
                continue
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)



