#!/bin/bash

# TimesNet M4 Dataset - Weekly and Daily Patterns Only
# This script trains TimesNet on M4 dataset for Weekly and Daily seasonal patterns

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimesNet

echo "Starting TimesNet training on M4 dataset..."
echo "Training patterns: Daily and Weekly"
echo "Checkpoint path: /content/drive/MyDrive/Model pths/time_series"

# Daily Pattern
echo "Training Daily pattern..."
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /content/drive/MyDrive/M4 \
  --checkpoints '/content/drive/MyDrive/Model pths/time_series' \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 16 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

echo "Daily pattern training completed!"
echo "---"

# Weekly Pattern
echo "Training Weekly pattern..."
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /content/drive/MyDrive/M4 \
  --checkpoints '/content/drive/MyDrive/Model pths/time_series' \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'

echo "Weekly pattern training completed!"
echo "---"
echo "All M4 training completed!"
