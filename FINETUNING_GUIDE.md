# Fine-tuning TimesNet: Complete Guide

## ✅ Fixed Issue

**Error**: `unrecognized arguments: --pretrained_checkpoint`

**Solution**: Added `--pretrained_checkpoint` argument to `run.py` argument parser (line 40)

---

## 🔧 Architecture Compatibility

### ⚠️ CRITICAL REQUIREMENT

When fine-tuning, the pretrained model and fine-tuning script **MUST have identical architecture**:

| Parameter | Value | Where to Check |
|-----------|-------|----------------|
| `d_model` | 384 | Look for `dm384` in checkpoint path |
| `n_heads` | 6 | Look for `nh6` in checkpoint path |
| `d_ff` | 1536 | Look for `df1536` in checkpoint path |
| `e_layers` | 2 | Look for `el2` in checkpoint path |
| `d_layers` | 1 | Look for `dl1` in checkpoint path |

### Example Checkpoint Path Breakdown

```
/content/drive/MyDrive/Model pths/time_series/M4_Weekly/
short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth
                                                          ^^^^^^^   ^^^      ^^  ^^  ^^^^^^^
                                                          d_model  n_heads   e   d    d_ff
                                                           =384     =6      =2  =1   =1536
```

### ❌ Incompatible Example

```
Pretrained: dm32_nh8_df32    ← d_model=32, n_heads=8, d_ff=32
Fine-tune:  dm384_nh6_df1536 ← d_model=384, n_heads=6, d_ff=1536

Result: ❌ WILL FAIL - shapes don't match!
```

### ✅ Compatible Example

```
Pretrained: dm384_nh6_df1536 ← d_model=384, n_heads=6, d_ff=1536
Fine-tune:  dm384_nh6_df1536 ← d_model=384, n_heads=6, d_ff=1536

Result: ✅ SUCCESS - shapes match!
```

---

## 🚀 Step-by-Step Fine-tuning

### Step 1: Train M4 Model (Optional)

```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_M4_Weekly_Daily.sh
```

**This will create checkpoints with architecture**: `d_model=384, n_heads=6, d_ff=1536`

**Checkpoint locations**:
- Daily: `/content/drive/MyDrive/Model pths/time_series/M4_Daily/short_term_forecast_m4_Daily_*/checkpoint.pth`
- Weekly: `/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_*/checkpoint.pth`

### Step 2: Find Your Checkpoint

After M4 training completes, you'll see output like:

```
Updating learning rate to 0.0001
Epoch: 7 cost time: 42.3 seconds
...
path:  /content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0
```

**Copy this path** and append `/checkpoint.pth`

### Step 3: Update Fine-tuning Script

Edit `TimesNet_total_finetune.sh` line 21:

```bash
PRETRAINED_CHECKPOINT='/content/drive/MyDrive/Model pths/time_series/M4_Weekly/short_term_forecast_m4_Weekly_TimesNet_m4_ftM_sl26_ll13_pl13_dm384_nh6_el2_dl1_df1536_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
```

**Verify the path contains**: `dm384_nh6_df1536`

### Step 4: Run Fine-tuning

```bash
cd Time-Series-Library
bash scripts/short_term_forecast/TimesNet_total_finetune.sh
```

You should see:

```
⚠️  ============================================
⚠️  ARCHITECTURE COMPATIBILITY CHECK
⚠️  ============================================
Fine-tuning script requires: d_model=384, n_heads=6, d_ff=1536
Pretrained checkpoint: /content/drive/MyDrive/Model pths/.../checkpoint.pth
Please verify the checkpoint path contains: dm384_nh6_df1536
⚠️  ============================================

Loading pretrained model from: /content/drive/MyDrive/Model pths/.../checkpoint.pth
✅ Pretrained model loaded successfully!
```

---

## 📊 Architecture Parameters

### Current Configuration (All Scripts)

All scripts now use **consistent architecture**:

```python
d_model = 384      # Model embedding dimension
d_ff = 1536        # Feed-forward dimension (4x d_model)
n_heads = 6        # Number of attention heads
e_layers = 2       # Number of encoder layers
d_layers = 1       # Number of decoder layers
```

### Scripts Using This Architecture

✅ `TimesNet_total.sh` - Train from scratch  
✅ `TimesNet_total_finetune.sh` - Fine-tune from M4  
✅ `TimesNet_M4_Weekly_Daily.sh` - M4 pretraining  

---

## 🔍 Troubleshooting

### Error: "unrecognized arguments: --pretrained_checkpoint"

**Solution**: Update to the latest code. The argument has been added to `run.py`.

### Error: Size mismatch when loading checkpoint

**Cause**: Architecture mismatch between pretrained and fine-tuning models.

**Solution**: 
1. Check checkpoint path for `dmXXX_nhXX_dfXXXX`
2. Verify it matches `dm384_nh6_df1536`
3. If not, either:
   - Find a different checkpoint with matching architecture
   - Or retrain M4 with `TimesNet_M4_Weekly_Daily.sh` (which uses 384/6/1536)

### Warning: "Pretrained checkpoint not found"

**Cause**: Checkpoint path is incorrect or file doesn't exist.

**Solution**: 
1. Check the file exists: `ls -lh /content/drive/MyDrive/Model\ pths/time_series/M4_Weekly/*/checkpoint.pth`
2. Copy the correct full path
3. Update `PRETRAINED_CHECKPOINT` variable in script

---

## 💡 Tips

1. **Always match architectures** - This is the #1 cause of fine-tuning failures
2. **Use lower learning rate** - Fine-tuning uses 0.0001 vs training's 0.001
3. **Fewer epochs needed** - Fine-tuning uses 20 epochs vs training's 50
4. **Organize checkpoints** - Use descriptive checkpoint directories
5. **Monitor early stopping** - Patience is 5 for fine-tuning vs 3 for training

---

## 📁 Checkpoint Organization

```
/content/drive/MyDrive/Model pths/time_series/
├── Battery/              # From-scratch training
├── Battery_Finetuned/    # Fine-tuned from M4
├── M4_Daily/            # M4 Daily pretraining
└── M4_Weekly/           # M4 Weekly pretraining (← use this for fine-tuning)
```

Choose **M4_Weekly** for battery fine-tuning as it has similar temporal patterns!

