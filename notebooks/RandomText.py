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


import matplotlib.font_manager as fm # to create font

from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random

# 随机字母:
def rndChar():
    return chr(66)  # chr(random.randint(65, 90))

# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
# font = ImageFont.truetype('Arial.ttf', 36)
fontsize = 36
font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=(0,0,0))  # 
# 模糊:
# image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg')


# In[ ]:





# In[ ]:





# In[ ]:


import matplotlib.font_manager as fm # to create font
from PIL import Image,ImageFont,ImageDraw
impath = '../exp/style_token_mi_beta1em2_notused-2/result_48k_random/Batch_0_rec_image.png'
textpath = '../exp/style_token_mi_beta1em2_notused-2/result_48k_random/Batch_0_labels.npy'


# In[ ]:


im = Image.open(impath)
im


# In[ ]:


text = np.load(textpath)
text


# In[ ]:


# 创建Font对象:
# font = ImageFont.truetype('Arial.ttf', 36)
fontsize = 10
font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)

shift = im.size[0]//8
draw = ImageDraw.Draw(im)
for i in range(8):
    for j in range(8):
        # print(text[i*8+j].item())
        draw.text(
            (i*shift+2, j*shift+1), 
            str(text[i+j*8].item()), 
            font=font, 
            fill=(256,128,128)
        )


# In[ ]:


im


# In[ ]:





# In[ ]:


def test_dict(config={}):
    config['a'] = 1
    print(config)
    
test_dict()


# In[ ]:


test_dict({'b':2})
test_dict()


# In[ ]:




