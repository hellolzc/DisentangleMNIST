import os, sys
import numpy as np

import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from dataset.mnist_m import MNISTM

from common.log_util import get_host_ip, get_hostname, get_cuda_version, get_python_version, log_summary
from common.model_summary import model_summary
from common.hparams import hparams_debug_string

from model.model_ed import ModelED
from model.loss import EncoderDecoderLoss


def prepare_val_dataloader(batch_size = 64, image_size = 28):
    ###################
    # load data       #
    ###################

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # gray2rgb_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3,1,1)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])   # 修改的位置

    # if name == 'mnist':
    #     mode = 'source'
    #     image_root = './data/'
    #     dataset = datasets.MNIST(
    #         root=image_root,
    #         train=False,
    #         transform=gray2rgb_transform
    #     )

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=8
    #     )

    # elif name == 'mnist_m':

    # mode = 'target'
    image_root = './data/'
    dataset = MNISTM(
        root=image_root,
        transform=img_transform,
        train=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    return dataloader



def prepare_dataloader(source_image_root, batch_size = 64, image_size = 28, ):
    #######################
    # load data           #
    #######################

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # gray2rgb_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3,1,1)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])   # 修改的位置

    # dataset_source = datasets.MNIST(
    #     root=source_image_root,
    #     train=True,
    #     transform=gray2rgb_transform
    # )

    # dataloader_source = torch.utils.data.DataLoader(
    #     dataset=dataset_source,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8
    # )

    dataset_target = MNISTM(
        root=source_image_root,
        train=True,
        transform=img_transform,
        download=False,
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return dataloader_target


def get_model(config):
    model_name = config['model']
    if model_name == 'ModelED':
        model = ModelED(config)
    elif model_name == 'ModelNTI':
        from model.model_ed import ModelNTI
        model = ModelNTI(config)
    elif model_name == 'ModelST':
        from model.model_st import ModelST
        model = ModelST(config)
    elif model_name == 'ModelSV':
        from model.model_st import ModelSV
        model = ModelSV(config)
    else:
        raise ValueError()

    return model

def get_loss_fn(config):
    model_name = config['model']
    if model_name in ['ModelED', 'ModelNTI', 'ModelST', 'ModelSV']:
        criterion = EncoderDecoderLoss()
    else:
        raise ValueError()
    return criterion

def log_exp(log_dir, model, config):
    log_summary(
        os.path.join(log_dir, "summary.log"),
        {
            '\nHost Name': get_hostname(),
            'Host IP': get_host_ip(),
            'Python Version': get_python_version(),
            'CUDA Version': get_cuda_version(),
            'PyTorch Version': torch.__version__,
            '\nModel': model_summary(model),
            '\nConfig': hparams_debug_string(config),
        }
    )


import torchvision.utils as vutils

def save_batch_results(output_dir, preffix, targets, predicts):
    input_img, class_label = targets[:2]
    ref_code, rec_img = predicts[:2]
    vutils.save_image(input_img, output_dir + '/%s_ori_image.png' % preffix, nrow=8)
    vutils.save_image(rec_img, output_dir + '/%s_rec_image.png' % preffix, nrow=8)
    np.save(output_dir + '/%s_labels' % preffix, class_label.data.cpu().numpy().squeeze())
    if len(predicts) > 2:
        weights, scores = predicts[2:]
        np.save(output_dir + '/%s_weights' % preffix, weights.data.cpu().numpy().squeeze())
        np.save(output_dir + '/%s_scores' % preffix, scores.data.cpu().numpy().squeeze())
