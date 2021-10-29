#!/bin/bash

#############################
# Encoder Decoder           #
#############################
EXP_NAME='encoder_decoder'

CUDA_VISIBLE_DEVICES=2 python train.py  --model 'ModelED' \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"


#############################
# Number To Image           #
#############################
EXP_NAME='number2image'

CUDA_VISIBLE_DEVICES=1 python train.py  --model 'ModelNTI' \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"



#############################
# Style Token               #
#############################
EXP_NAME='style_token'

CUDA_VISIBLE_DEVICES=0 python train.py  --model 'ModelST' \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"



#######################################
# Style Variation Bernoulli           #
#######################################
EXP_NAME='style_variation_ber'

CUDA_VISIBLE_DEVICES=0 python train.py  --model 'ModelSVB' \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"

#######################################
# Style Variation Categorical         #
#######################################
EXP_NAME='style_variation_cat'

CUDA_VISIBLE_DEVICES=0 python train.py  --model 'ModelSVB' \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"
