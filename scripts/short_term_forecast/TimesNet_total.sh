#!/bin/bash

# TimesNet Battery Lifecycle Prediction Script
# Based on Lag-Llama incremental training parameters
# This script trains TimesNet on battery data with the following split:
# - Training: First 70% of cycles from b1c0_for_model.csv
# - Validation: Last 30% of cycles from b1c0_for_model.csv  
# - Test: All cycles from other CSV files (excluding b1c0)

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimesNet

echo "Starting TimesNet training on battery data..."
echo "Training data: b1c0_for_model.csv (70% for train, 30% for val)"
echo "Test data: All other CSV files in ../../total/"
echo "Features: charge_capacity, discharge_capacity, internal_resistance, temperature_mean, temperature_min, temperature_max"
echo "Using Lag-Llama optimized parameters..."

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /content/drive/MyDrive/total \
  --checkpoints '/content/drive/MyDrive/Model pths/time_series' \
  --model_id battery_total_lagllama_params \
  --model $model_name \
  --data battery \
  --features MS \
  --target discharge_capacity \
  --seq_len 24 \
  --label_len 8 \
  --pred_len 8 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --batch_size 4 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --top_k 5 \
  --des 'Battery_Forecast_LagLlamaParams' \
  --itr 1 \
  --train_epochs 50 \
  --learning_rate 0.001 \
  --loss MSE \
  --freq d \
  --embed timeF \
  --patience 3 \
  --dropout 0.1 \
  --lradj type1 \
  --use_amp

echo "Training completed!"
