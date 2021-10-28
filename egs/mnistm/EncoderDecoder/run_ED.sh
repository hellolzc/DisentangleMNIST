#!/bin/bash

MAIN_ROOT=$(cd ../../.. && pwd)

# NOTE(hellolzc): Allow running as "python -m espnet2.bin.xxx"
export PYTHONPATH=$MAIN_ROOT

conda activate tf1x


CUDA_VISIBLE_DEVICES=2 python "${MAIN_ROOT}"/train.py

