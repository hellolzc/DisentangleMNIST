#!/bin/bash

#############################
# Style Token               #
#############################
EXP_NAME='style_token'
read -r -d '' EXTRA_HP <<- EOF
model: ModelST
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 5
EOF

EXP_NAME='style_token_mi_loss'
read -r -d '' EXTRA_HP <<- EOF
model: ModelST
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 5
loss_weight_rec: 1.0
loss_weight_mi: 1.0e-6
use_mi: True
mi_iters: 5
EOF

#######################################
# Style Variation Bernoulli           #
#######################################
EXP_NAME='style_variation_ber_g1s5k'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 5
gumbel_activation: 'sigmoid'
gumbel_start_step: 5000
EOF



#######################################
# Style Variation Categorical         #
#######################################
EXP_NAME='style_variation_cat_g1s5k'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 5
gumbel_activation: 'softmax'
gumbel_start_step: 5000
EOF




#######################################
# Train                               #
#######################################
printf "EXP_NAME: %s\nEXTRA_HP:\n%s\n" "$EXP_NAME" "$EXTRA_HP"

export CUDA_VISIBLE_DEVICES=0
python train_mi.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"
