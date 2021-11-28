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


import torch
import torch.backends.cudnn as cudnn

from utils import get_model, get_loss_fn
from common.hparams import create_hparams

from common.device_funcs import to_device
from utils import prepare_val_dataloader, save_batch_results


# In[ ]:


EXP_NAME='style_token'
EXTRA_HP='''
model: ModelST
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 10
'''
config = create_hparams(yaml_hparams_string=EXTRA_HP, allow_add=True)


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True
cudnn.benchmark = True


# In[ ]:


class args():
    ckpt_dir = f"../exp/{EXP_NAME}_mi_adam/ckpt" 
    log_dir = f"../exp/{EXP_NAME}_mi_adam/log"
    load_model = f"../exp/{EXP_NAME}/ckpt/sv_mnist_48.pth"

my_net = get_model(config)
checkpoint = torch.load(args.load_model)
my_net.load_state_dict(checkpoint)
my_net.eval()

my_net = my_net.to(device)


# In[ ]:


from model.mi_estimators import CLUBSample
from utils import prepare_dataloader, get_model, get_loss_fn, log_exp

def get_mi_net(config):
    code_size=config['code_size']
    mi_net = CLUBSample(x_dim=code_size, y_dim=code_size, hidden_size=256)
    return mi_net


def mi_forward(model, mi_net, data_target):

    input_img, class_label = data_target

    with torch.no_grad():
        weights, scores, style_embs, text_embs = model.encode(input_data=input_img, number=class_label)

    x = style_embs.detach()
    y = text_embs.detach()

    import pdb; pdb.set_trace()
    lld_loss = mi_net.negative_loglikeli(x, y)

    loss = lld_loss

    with torch.no_grad():
        mi_est = mi_net.mi_est(x, y)
    loss_dict = {
        'lld_loss': lld_loss,
        'mi_est': mi_est,
    }

    return loss, loss_dict


# In[ ]:


image_root = os.path.join('./data/')
image_size=28

n_epoch = 1


my_net = my_net.to(device)
mi_net = get_mi_net(config).to(device)

mi_checkpoint = torch.load(f"../exp/{EXP_NAME}_mi_adam/ckpt/mi_net_48.pth")
mi_net.load_state_dict(mi_checkpoint)
mi_net.eval()



#############################
# iter     network          #
#############################
dataloader = prepare_dataloader(image_root, batch_size=64, image_size=image_size)

len_dataloader = len(dataloader)

current_step = 0
for epoch in range(n_epoch):
    data_iter = iter(dataloader)
    for i in range(len_dataloader):

        data_target = data_iter.next()
        data_target = to_device(data_target, device=device)

        # TODO: fix this bug
        if config["model"] in ['ModelSV',]:
            assert False, 'Not support now'
            my_net.set_step(current_step)

        loss, loss_dict = mi_forward(my_net, mi_net, data_target)

        break


# In[ ]:


loss, loss_dict


# In[ ]:


# %pdb


# In[ ]:





# In[ ]:




