#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import List, Tuple
import os, sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import librosa
import librosa.display
import IPython.display

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sys.path.append('..')


# In[ ]:


from utils import get_model
from common.hparams import create_hparams


# In[ ]:


EXP_NAME='style_token_diff_loss'
EXTRA_HP='''
model: ModelST
loss_fn: StyleDiffLoss
code_size: 128
n_class: 10
token_num: 5
'''
config = create_hparams(yaml_hparams_string=EXTRA_HP, allow_add=True)


# In[ ]:


import torch

epoch=48
my_net = get_model(config)
checkpoint = torch.load(os.path.join('../exp/'+EXP_NAME+ '/ckpt', 'sv_mnist_' + str(epoch) + '.pth'))
my_net.load_state_dict(checkpoint)
my_net.eval()


# In[ ]:


gst_embs = my_net.style_encoder.stl.gst_embs.detach()


# In[ ]:


gst_embs.max()


# In[ ]:





# In[ ]:


import os
import torch
import torch.backends.cudnn as cudnn


from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def test(model, criterion, epoch, step, 
        name='MINIST-M',
        logger=None,
        log_dir = './log/',
    ):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True

    dataloader = prepare_val_dataloader()

    model.eval()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    with torch.no_grad():
        n_total = 0
        total_loss_sum = 0.0
        loss_dict_sum = {}
        for i in  range(len_dataloader):
            data_input = data_iter.next()
            data_input = to_device(data_input, device=device)
            input_img, class_label = data_input
            batch_size = len(class_label)

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

            if i == len_dataloader - 2:
                save_batch_results(log_dir, 'Epoch_%d' % epoch, data_input, result)

            n_total += batch_size

    loss_mean = total_loss_sum / n_total
    loss_dict_mean = {
        k: v / n_total for k, v in loss_dict_sum.items()
    }
    print(
        'Step: %d, Epoch: %d, Validation on %d samples. total_loss: %.6f, ' % (step, epoch, n_total, loss_mean),
        ' '.join(['%s: %.6f' % (k,v) for k, v in loss_dict_mean.items()]),
    )

    if logger is not None:
        logger.log_validation(loss_mean, model, step, scalar_dict=loss_dict)

    model.train()


# In[ ]:


from utils import get_loss_fn
criterion = get_loss_fn(config).to(device)


# In[ ]:


my_net = my_net.to(device)
test(my_net, criterion, 48, 0)


# In[ ]:


criterion


# In[ ]:


# !ln -sT ../data/ data
# !mkdir log


# In[ ]:





# In[ ]:





# In[ ]:




