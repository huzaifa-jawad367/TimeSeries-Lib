# Quick Start Guide - Battery Fine-tuning

## ✅ Issue Fixed
Added `--pretrained_checkpoint` argument to `run.py`. Fine-tuning now works!

---

## 🚀 Option 1: Train from Scratch (No Pretrained Model)

```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_total.sh
```

**Time**: ~30 minutes  
**Checkpoint**: `/content/drive/MyDrive/Model pths/time_series/Battery/`

---

## 🚀 Option 2: Fine-tune from M4 (Recommended)

### A. First, train M4 (one-time setup)

```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_M4_Weekly_Daily.sh
```

**Time**: ~30 minutes  
**Checkpoint**: `/content/drive/MyDrive/Model pths/time_series/M4_Weekly/`

### B. Copy the checkpoint path from output

Look for this in the output:
```
path: /content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0
```

**Full checkpoint path**:
```
/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth
```

### C. Edit fine-tuning script

```bash
nano scripts/short_term_forecast/TimesNet_total_finetune.sh
```

Update line 21:
```bash
PRETRAINED_CHECKPOINT='/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
```

### D. Run fine-tuning

```bash
bash scripts/short_term_forecast/TimesNet_total_finetune.sh
```

**Time**: ~15 minutes  
**Checkpoint**: `/content/drive/MyDrive/Model pths/time_series/Battery_Finetuned/`

---

## ⚠️ Architecture Must Match!

The checkpoint path MUST contain:
- `dm384` (d_model=384)
- `nh6` (n_heads=6)
- `df1536` (d_ff=1536)

**✅ Correct**: `dm384_nh6_el2_dl1_df1536`  
**❌ Wrong**: `dm32_nh8_el2_dl1_df32` ← Different architecture, won't work!

---

## 📂 Data Setup

Your data should be in `/content/drive/MyDrive/total/`:

```
total/
├── b1c0_for_model.csv    ← Training & Validation
├── b1c1_for_model.csv    ← Test
├── b1c2_for_model.csv    ← Test
├── ...
└── b7c10_for_model.csv   ← Test
```

**Split**:
- Training: First 70% of b1c0
- Validation: Last 30% of b1c0
- Test: All other files

---

## 📊 Expected Output

```
⚠️  ============================================
⚠️  ARCHITECTURE COMPATIBILITY CHECK
⚠️  ============================================
Fine-tuning script requires: d_model=384, n_heads=6, d_ff=1536
Pretrained checkpoint: /content/drive/.../checkpoint.pth
Please verify the checkpoint path contains: dm384_nh6_df1536
⚠️  ============================================

Loading pretrained model from: /content/drive/.../checkpoint.pth
✅ Pretrained model loaded successfully!
Epoch: 1 cost time: 12.5 seconds
...
```

---

## 🎯 Which Option to Choose?

| Option | Use When | Advantage |
|--------|----------|-----------|
| **From Scratch** | First time, no M4 model | Simple, one script |
| **Fine-tuning** | Have M4 model | Better performance, faster convergence |

**Recommendation**: Try fine-tuning if you have time for M4 training first!

