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
token_num: 10
EOF

EXP_NAME='style_token_mi_beta1em2'
read -r -d '' EXTRA_HP <<- EOF
model: ModelST
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 10
loss_weight_rec: 1.0
loss_weight_mi: 1.0e-2
use_mi: True  # add mi_loss to total_loss
mi_iters: 5
mi_start_step = 8000
adv_choose='PlanA'
EOF

EXP_NAME='style_token_mi_beta1em2_notused'
read -r -d '' EXTRA_HP <<- EOF
model: ModelST
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 10
loss_weight_rec: 1.0
loss_weight_mi: 1.0e-2
use_mi: False  # add mi_loss to total_loss
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
token_num: 10
gumbel_activation: 'sigmoid'
gumbel_start_step: 5000
EOF


EXP_NAME='style_variation_ber_g1s5k_mi_beta1em2'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 10
gumbel_activation: 'sigmoid'
gumbel_start_step: 5000
loss_weight_rec: 1.0
loss_weight_mi: 1.0e-2
use_mi: False  # add mi_loss to total_loss
mi_iters: 5
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
token_num: 10
gumbel_activation: 'softmax'
gumbel_start_step: 5000
EOF


EXP_NAME='style_variation_cat_g1s5k_mi_beta1em2'
read -r -d '' EXTRA_HP <<- EOF
model: ModelSV
loss_fn: EncoderDecoderLoss
code_size: 128
n_class: 10
token_num: 10
gumbel_activation: 'softmax'
gumbel_start_step: 5000
loss_weight_rec: 1.0
loss_weight_mi: 1.0e-2
use_mi: False  # add mi_loss to total_loss
mi_iters: 5
EOF



#######################################
# Train                               #
#######################################
printf "EXP_NAME: %s\nEXTRA_HP:\n%s\n" "$EXP_NAME" "$EXTRA_HP"

export CUDA_VISIBLE_DEVICES=0
python train_with_mi.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --log_dir "./exp/${EXP_NAME}/log"


#######################################
# Infer                               #
#######################################
python infer.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --epoch 48 --out_dir "./exp/${EXP_NAME}/result_48k"

python infer.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}/ckpt" --epoch 48 --out_dir "./exp/${EXP_NAME}/result_48k_random"  --random_text


#######################################
# Check                               #
#######################################
python train_clf.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}_clf/ckpt" --log_dir "./exp/${EXP_NAME}_clf/log" \
    --load_model "./exp/${EXP_NAME}/ckpt/sv_mnist_48.pth"


python train_mi.py  --hparams "$EXTRA_HP" \
    --ckpt_dir "./exp/${EXP_NAME}_mi_adam/ckpt" --log_dir "./exp/${EXP_NAME}_mi_adam/log" \
    --load_model "./exp/${EXP_NAME}/ckpt/sv_mnist_48.pth"

