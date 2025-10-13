#!/bin/bash

# TimesNet Battery Lifecycle Prediction - Fine-tuning from Pretrained Model
# This script fine-tunes a pretrained TimesNet model on battery data
# Training split:
# - Training: First 70% of cycles from b1c0_for_model.csv
# - Validation: Last 30% of cycles from b1c0_for_model.csv  
# - Test: All cycles from other CSV files (excluding b1c0)

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name=TimesNet

# ⚠️  IMPORTANT: Set the path to your pretrained checkpoint here
# ⚠️  The checkpoint MUST have matching architecture:
# ⚠️  - d_model=384, n_heads=6, d_ff=1536 (as specified below)
# ⚠️  Look for checkpoint path containing: dm384_nh6_el2_dl1_df1536
# 
# Example: '/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
PRETRAINED_CHECKPOINT='/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'

echo "⚠️  ============================================"
echo "⚠️  ARCHITECTURE COMPATIBILITY CHECK"
echo "⚠️  ============================================"
echo "Fine-tuning script requires: d_model=384, n_heads=6, d_ff=1536"
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Please verify the checkpoint path contains: dm384_nh6_df1536"
echo "If not, the model loading will FAIL!"
echo "⚠️  ============================================"
echo ""
echo "Starting TimesNet fine-tuning on battery data..."
echo "Training data: b1c0_for_model.csv (70% for train, 30% for val)"
echo "Test data: All other CSV files in total/"
echo "Features: charge_capacity, discharge_capacity, internal_resistance, temperature_mean, temperature_min, temperature_max"

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /content/drive/MyDrive/total \
  --checkpoints '/content/drive/MyDrive/Model pths/time_series/Battery_Finetuned' \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --model_id battery_finetuned \
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
  --d_model 384 \
  --d_ff 1536 \
  --n_heads 6 \
  --top_k 5 \
  --des 'Battery_Finetuned_From_M4' \
  --itr 1 \
  --train_epochs 20 \
  --learning_rate 0.0001 \
  --loss MSE \
  --freq d \
  --embed timeF \
  --patience 5 \
  --dropout 0.1 \
  --lradj type1 \
  --use_amp

echo "Fine-tuning completed!"

