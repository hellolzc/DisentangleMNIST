#!/usr/bin/env python
# coding: utf-8
import os
import argparse

import torch
import torch.backends.cudnn as cudnn

from common.hparams import create_hparams
from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results
from utils import get_model, get_loss_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def infer(
    model,
    criterion,
    dataloader,
    out_dir = './result/',
):

    ###################
    # params          #
    ###################
    os.makedirs(out_dir, exist_ok=True)
    cuda = True
    cudnn.benchmark = True

    model.eval()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    with torch.no_grad():
        n_total = 0
        total_loss_sum = 0.0
        loss_dict_sum = {}
        for i in  range(len_dataloader):
            data_input = data_iter.next()
            input_img, class_label = data_input
            batch_size = len(class_label)

            input_img = to_device(input_img, device=device)
            class_label = to_device(class_label, device=device)

            result = model(input_data=input_img, number=class_label)
            loss, loss_dict = criterion(data_input, result)

            total_loss_sum += loss * batch_size
            if loss_dict_sum is None:
                loss_dict_sum = {
                    k: v.item() * batch_size for k, v in loss_dict.items()
                }
            else:
                for k in loss_dict_sum:
                    loss_dict_sum[k] += loss_dict[k].item() * batch_size

            # if i == len_dataloader - 2:
            save_batch_results(out_dir, 'Batch_%d' % i, data_input, result)

            n_total += batch_size

    loss_mean = total_loss_sum / n_total
    loss_dict_mean = {
        k: v / n_total for k, v in loss_dict_sum.items()
    }
    print(
        'Validation on %d samples. total_loss: %.6f, ' % (n_total, loss_mean),
        ' '.join(['%s: %.6f' % (k,v) for k, v in loss_dict_mean.items()]),
    )

    model.train()





if __name__ == '__main__':
    ######################
    # params             #
    ######################
    parser = argparse.ArgumentParser(
        description="Infer Disentangle (See detail in train.py)."
    )
    parser.add_argument("--ckpt_dir", type=str, default="exp/untitled/ckpt",
        required=False, help="Directory to save checkpoint"
    )
    parser.add_argument("--epoch", type=int, default=48,
        required=False, help="which model to load"
    )
    parser.add_argument("--out_dir", type=str, default="exp/untitled/out",
        required=False, help="Directory to save outputs"
    )
    parser.add_argument("--hparams", type=str, default="",
        required=False, help="yaml style dict to update config"
    )

    args = parser.parse_args()
    config = create_hparams(
        yaml_hparams_string=args.hparams,
        debug_print=True,
        allow_add=True,
    )

    criterion = get_loss_fn(config).to(device)
    my_net = get_model(config)
    checkpoint = torch.load(os.path.join(args.ckpt_dir, 'sv_mnist_' + str(args.epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    my_net = my_net.to(device)

    dataloader = prepare_val_dataloader()

    infer(my_net, criterion, dataloader, args.out_dir)

