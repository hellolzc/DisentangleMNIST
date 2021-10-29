#!/bin/bash

#############################
# Encoder Decoder           #
#############################
EXP_NAME='encoder_decoder'
# EXTRA_HP=''
read -r -d '' EXTRA_HP <<- EOF
model: ModelED
code_size: 128
n_class: 10
EOF


#############################
# Number To Image           #
#############################
EXP_NAME='number2image'
read -r -d '' EXTRA_HP <<- EOF
model: ModelNTI
code_size: 128
n_class: 10
EOF


#############################
# Style Token               #
#############################
EXP_NAME='style_token'
read -r -d '' EXTRA_HP <<- EOF
model: ModelST
code_size: 128
n_class: 10
token_num: 5
EOF

#######################################
# Style Variation Bernoulli           #
#######################################
EXP_NAME='style_variation_ber'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
code_size: 128
n_class: 10
token_num: 5
gumbel_activation: 'sigmoid'
EOF

#######################################
# Style Variation Categorical         #
#######################################
EXP_NAME='style_variation_cat'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
code_size: 128
n_class: 10
token_num: 5
gumbel_activation: 'softmax'
EOF


#######################################
# Train                               #
#######################################
printf "EXP_NAME: %s\nEXTRA_HP:\n%s\n" "$EXP_NAME" "$EXTRA_HP"

export CUDA_VISIBLE_DEVICES=2
python train.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"
