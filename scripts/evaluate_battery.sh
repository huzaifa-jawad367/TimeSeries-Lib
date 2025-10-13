#!/bin/bash

# Battery Model Evaluation Script
# Evaluates a pretrained TimesNet checkpoint on the battery test set
# Calculates R², RMSE, and MAE metrics

export CUDA_VISIBLE_DEVICES=0

# ⚠️  IMPORTANT: Set the path to your trained checkpoint here
# This should be the checkpoint.pth file from your training run
CHECKPOINT_PATH='/content/drive/MyDrive/Model pths/time_series/Battery/short_term_forecast_battery_total_lagllama_params_TimesNet_battery_ftMS_sl24_ll8_pl8_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'

# Data and result paths
ROOT_PATH='/content/drive/MyDrive/total'
RESULT_PATH='/content/drive/MyDrive/evaluation_results'
MODEL_ID='battery_timesnet_evaluation'

echo "============================================"
echo "Battery Model Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data path: $ROOT_PATH"
echo "Results will be saved to: $RESULT_PATH/$MODEL_ID"
echo "============================================"
echo ""

python -u evaluate_battery.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --result_path "$RESULT_PATH" \
  --model_id "$MODEL_ID" \
  --root_path "$ROOT_PATH" \
  --data battery \
  --model TimesNet \
  --features MS \
  --target discharge_capacity \
  --seq_len 24 \
  --label_len 8 \
  --pred_len 8 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 384 \
  --d_ff 1536 \
  --n_heads 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --top_k 5 \
  --freq d \
  --embed timeF \
  --batch_size 32 \
  --num_workers 4 \
  --save_predictions True \
  --save_plots True \
  --use_amp

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: $RESULT_PATH/$MODEL_ID"
echo "============================================"

