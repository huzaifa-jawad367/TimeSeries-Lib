# Battery Training Scripts Guide

## Overview
This directory contains scripts for training TimesNet on battery lifecycle prediction.

## Available Scripts

### 1. TimesNet_total.sh
**Purpose**: Train TimesNet from scratch on battery data  
**Model**: d_model=384, d_ff=1536, n_heads=6  
**Training**: 50 epochs with early stopping (patience=3)  
**Data Split**:
- Training: First 70% of b1c0_for_model.csv
- Validation: Last 30% of b1c0_for_model.csv
- Test: All other CSV files in total/ (123 batteries)

**Usage**:
```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_total.sh
```

**Checkpoint**: `/content/drive/MyDrive/Model pths/time_series/Battery/`

---

### 2. TimesNet_total_finetune.sh
**Purpose**: Fine-tune a pretrained TimesNet model on battery data  
**Model**: d_model=384, d_ff=1536, n_heads=6 (must match pretrained model)  
**Training**: 20 epochs with early stopping (patience=5)  
**Learning Rate**: 0.0001 (10x lower for fine-tuning)

**Before Running**:
1. Edit the script and set `PRETRAINED_CHECKPOINT` variable to your checkpoint path
2. Example: 
   ```bash
   PRETRAINED_CHECKPOINT='/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_.../checkpoint.pth'
   ```

**Usage**:
```bash
cd Time-Series-Library
# Edit the script first to set PRETRAINED_CHECKPOINT
bash scripts/short_term_forecast/TimesNet_total_finetune.sh
```

**Checkpoint**: `/content/drive/MyDrive/Model pths/time_series/Battery_Finetuned/`

---

### 3. TimesNet_M4_Weekly_Daily.sh
**Purpose**: Train TimesNet on M4 dataset (Weekly and Daily patterns)  
**Model**: d_model=384, d_ff=1536, n_heads=6  
**Data**: M4 Weekly (359 series) and Daily (4,227 series)

**Usage**:
```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_M4_Weekly_Daily.sh
```

**Checkpoints**:
- Daily: `/content/drive/MyDrive/Model pths/time_series/M4_Daily/`
- Weekly: `/content/drive/MyDrive/Model pths/time_series/M4_Weekly/`

---

## Model Parameters Explanation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 384 | Model embedding dimension |
| `d_ff` | 1536 | Feed-forward dimension (4x d_model) |
| `n_heads` | 6 | Number of attention heads |
| `seq_len` | 24 | Input sequence length (cycles) |
| `pred_len` | 8 | Prediction horizon (cycles) |
| `batch_size` | 4 | Batch size for training |
| `train_epochs` | 50/20 | Maximum training epochs |
| `patience` | 3/5 | Early stopping patience |

---

## Training Workflow

### Option A: Train from Scratch
```bash
# Step 1: Train on battery data from random initialization
bash scripts/short_term_forecast/TimesNet_total.sh

# Result: Model saved in Battery/ directory
```

### Option B: Fine-tune from M4
```bash
# Step 1: Train on M4 dataset first
bash scripts/short_term_forecast/TimesNet_M4_Weekly_Daily.sh

# Step 2: Copy the checkpoint path from M4_Weekly or M4_Daily

# Step 3: Edit TimesNet_total_finetune.sh with the checkpoint path

# Step 4: Fine-tune on battery data
bash scripts/short_term_forecast/TimesNet_total_finetune.sh

# Result: Fine-tuned model saved in Battery_Finetuned/ directory
```

---

## Expected Training Times (T4 GPU)

| Script | Time | Notes |
|--------|------|-------|
| `TimesNet_total.sh` | 20-40 min | From scratch, may stop early |
| `TimesNet_total_finetune.sh` | 10-20 min | Fine-tuning is faster |
| `TimesNet_M4_Weekly_Daily.sh` | 20-40 min | Both Daily + Weekly |

---

## Checkpoint Structure

```
/content/drive/MyDrive/Model pths/time_series/
├── Battery/                    # From-scratch battery model
│   └── short_term_forecast_battery_total_*/
│       └── checkpoint.pth
├── Battery_Finetuned/         # Fine-tuned from M4
│   └── short_term_forecast_battery_finetuned_*/
│       └── checkpoint.pth
├── M4_Daily/                  # M4 Daily pattern
│   └── short_term_forecast_m4_Daily_*/
│       └── checkpoint.pth
└── M4_Weekly/                 # M4 Weekly pattern
    └── short_term_forecast_m4_Weekly_*/
        └── checkpoint.pth
```

---

## Notes

- All models use the same architecture (d_model=384, d_ff=1536, n_heads=6)
- Fine-tuning requires matching architecture between pretrained and target model
- Checkpoints are automatically saved to Google Drive
- Early stopping prevents overfitting

