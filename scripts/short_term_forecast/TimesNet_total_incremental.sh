#!/bin/bash

# TimesNet Battery Lifecycle Prediction Script - Incremental Training
# Based on Lag-Llama incremental training approach with epoch decay
# This script implements the same incremental training strategy as Lag-Llama

export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

# Lag-Llama incremental training parameters
initial_epochs=5
epoch_decay=0.95
learning_rate=0.001

echo "Starting TimesNet incremental training on battery data..."
echo "Initial epochs: $initial_epochs"
echo "Epoch decay: $epoch_decay"
echo "Learning rate: $learning_rate"
echo "Training data: b1c0_for_model.csv (70% for train, 30% for val)"
echo "Test data: All other CSV files in ../../total/"

# Run multiple training iterations with decaying epochs (simulating incremental training)
for i in {0..4}; do
    current_epochs=$((initial_epochs - i))
    current_epochs=$((current_epochs > 2 ? current_epochs : 2))  # Minimum 2 epochs
    
    echo "Iteration $((i+1))/5: Training with $current_epochs epochs"
    
    python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ../../total \
      --model_id battery_total_incremental_iter${i} \
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
      --batch_size 64 \
      --d_model 512 \
      --d_ff 2048 \
      --n_heads 8 \
      --top_k 5 \
      --des "Battery_Forecast_Incremental_Iter${i}" \
      --itr 1 \
      --train_epochs $current_epochs \
      --learning_rate $learning_rate \
      --loss MSE \
      --freq d \
      --embed timeF \
      --patience 2 \
      --dropout 0.1 \
      --lradj type1
    
    echo "Completed iteration $((i+1)) with $current_epochs epochs"
    echo "---"
done

echo "Incremental training completed!"
