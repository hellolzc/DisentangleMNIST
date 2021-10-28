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


#############################
# Style Variation           #
#############################
