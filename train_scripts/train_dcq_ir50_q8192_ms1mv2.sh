#!/usr/bin/env bash
set -x

save_dir=Logs/dcq_ires50_q8192_ms1mv2
resume_dir=
mkdir -p $save_dir
echo '=> Log and checkpoint will be saved to '$save_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
    -a iresnet50 \
    --feat-dim 512 \
    --scale 50 \
    --margin 0.3 \
    --queue-size 8192 \
    --lr 0.06 \
    --batch-size 128 \
    --iter-per-epoch 11000 \
    --schedule 8 16 18 \
    --epochs 20 \
    --dataprob "1" \
    --filelist "ms1mv2.json" \
    --distributed \
    "DCQ_train_test_data/MS1MV2" $save_dir
