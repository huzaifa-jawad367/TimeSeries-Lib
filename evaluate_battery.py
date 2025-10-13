"""
Battery Model Evaluation Script
Evaluates a pretrained TimesNet model on battery test data using R², RMSE, and MAE metrics
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.tools import string_split


def calculate_metrics(pred, true):
    """
    Calculate R², RMSE, and MAE metrics
    
    Args:
        pred: numpy array of predictions, shape (samples, pred_len, features)
        true: numpy array of ground truth, shape (samples, pred_len, features)
    
    Returns:
        dict: Dictionary containing metrics for each feature and overall
    """
    metrics = {}
    
    # Flatten for overall metrics
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    
    # Overall metrics
    metrics['overall'] = {
        'R2': r2_score(true_flat, pred_flat),
        'RMSE': np.sqrt(mean_squared_error(true_flat, pred_flat)),
        'MAE': mean_absolute_error(true_flat, pred_flat)
    }
    
    # Per-feature metrics (if multivariate)
    if len(pred.shape) == 3 and pred.shape[2] > 1:
        feature_names = ['charge_capacity', 'discharge_capacity', 'internal_resistance', 
                        'temperature_mean', 'temperature_min', 'temperature_max']
        
        for i in range(pred.shape[2]):
            pred_feat = pred[:, :, i].reshape(-1)
            true_feat = true[:, :, i].reshape(-1)
            
            feat_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            metrics[feat_name] = {
                'R2': r2_score(true_feat, pred_feat),
                'RMSE': np.sqrt(mean_squared_error(true_feat, pred_feat)),
                'MAE': mean_absolute_error(true_feat, pred_feat)
            }
    
    # Per-timestep metrics
    metrics['per_timestep'] = {}
    for t in range(pred.shape[1]):
        pred_t = pred[:, t, :].reshape(-1)
        true_t = true[:, t, :].reshape(-1)
        
        metrics['per_timestep'][f't+{t+1}'] = {
            'R2': r2_score(true_t, pred_t),
            'RMSE': np.sqrt(mean_squared_error(true_t, pred_t)),
            'MAE': mean_absolute_error(true_t, pred_t)
        }
    
    return metrics


def save_predictions(preds, trues, save_path, dataset):
    """
    Save predictions and ground truth to CSV files
    
    Args:
        preds: numpy array of predictions
        trues: numpy array of ground truth
        save_path: directory to save files
        dataset: dataset object with inverse_transform method
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Inverse transform if scaler is available
    if hasattr(dataset, 'scaler') and dataset.scale:
        # Reshape for inverse transform
        batch_size, pred_len, n_features = preds.shape
        preds_reshaped = preds.reshape(-1, n_features)
        trues_reshaped = trues.reshape(-1, n_features)
        
        preds_inv = dataset.inverse_transform(preds_reshaped).reshape(batch_size, pred_len, n_features)
        trues_inv = dataset.inverse_transform(trues_reshaped).reshape(batch_size, pred_len, n_features)
    else:
        preds_inv = preds
        trues_inv = trues
    
    # Save to CSV
    feature_names = ['charge_capacity', 'discharge_capacity', 'internal_resistance', 
                    'temperature_mean', 'temperature_min', 'temperature_max']
    
    for i, feat in enumerate(feature_names[:preds.shape[2]]):
        # Predictions
        pred_df = pd.DataFrame(
            preds_inv[:, :, i],
            columns=[f't+{j+1}' for j in range(preds.shape[1])]
        )
        pred_df.to_csv(os.path.join(save_path, f'predictions_{feat}.csv'), index=False)
        
        # Ground truth
        true_df = pd.DataFrame(
            trues_inv[:, :, i],
            columns=[f't+{j+1}' for j in range(trues.shape[1])]
        )
        true_df.to_csv(os.path.join(save_path, f'ground_truth_{feat}.csv'), index=False)
    
    print(f'✅ Predictions saved to {save_path}')


def plot_results(preds, trues, metrics, save_path, dataset):
    """
    Create visualization plots for evaluation results
    
    Args:
        preds: numpy array of predictions
        trues: numpy array of ground truth
        metrics: dictionary of calculated metrics
        save_path: directory to save plots
        dataset: dataset object with inverse_transform method
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Inverse transform if scaler is available
    if hasattr(dataset, 'scaler') and dataset.scale:
        batch_size, pred_len, n_features = preds.shape
        preds_reshaped = preds.reshape(-1, n_features)
        trues_reshaped = trues.reshape(-1, n_features)
        
        preds_inv = dataset.inverse_transform(preds_reshaped).reshape(batch_size, pred_len, n_features)
        trues_inv = dataset.inverse_transform(trues_reshaped).reshape(batch_size, pred_len, n_features)
    else:
        preds_inv = preds
        trues_inv = trues
    
    feature_names = ['charge_capacity', 'discharge_capacity', 'internal_resistance', 
                    'temperature_mean', 'temperature_min', 'temperature_max']
    
    # 1. Scatter plots for each feature
    n_features = min(preds.shape[2], len(feature_names))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_features):
        pred_feat = preds_inv[:, :, i].reshape(-1)
        true_feat = trues_inv[:, :, i].reshape(-1)
        
        axes[i].scatter(true_feat, pred_feat, alpha=0.5, s=10)
        axes[i].plot([true_feat.min(), true_feat.max()], 
                     [true_feat.min(), true_feat.max()], 
                     'r--', lw=2, label='Perfect prediction')
        
        feat_name = feature_names[i]
        r2 = metrics.get(feat_name, {}).get('R2', 0)
        rmse = metrics.get(feat_name, {}).get('RMSE', 0)
        mae = metrics.get(feat_name, {}).get('MAE', 0)
        
        axes[i].set_xlabel('Ground Truth', fontsize=12)
        axes[i].set_ylabel('Predictions', fontsize=12)
        axes[i].set_title(f'{feat_name}\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}', fontsize=11)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time series examples
    n_examples = min(10, preds.shape[0])
    fig, axes = plt.subplots(n_examples, 1, figsize=(15, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(n_examples):
        # Plot discharge capacity (most important feature)
        discharge_idx = 1  # discharge_capacity index
        axes[i].plot(range(preds.shape[1]), trues_inv[i, :, discharge_idx], 
                    'b-o', label='Ground Truth', linewidth=2, markersize=6)
        axes[i].plot(range(preds.shape[1]), preds_inv[i, :, discharge_idx], 
                    'r--s', label='Prediction', linewidth=2, markersize=6)
        axes[i].set_xlabel('Prediction Horizon (cycles)', fontsize=11)
        axes[i].set_ylabel('Discharge Capacity', fontsize=11)
        axes[i].set_title(f'Example {i+1}', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'time_series_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics per timestep
    timesteps = list(metrics['per_timestep'].keys())
    r2_values = [metrics['per_timestep'][t]['R2'] for t in timesteps]
    rmse_values = [metrics['per_timestep'][t]['RMSE'] for t in timesteps]
    mae_values = [metrics['per_timestep'][t]['MAE'] for t in timesteps]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(range(len(timesteps)), r2_values, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score vs Prediction Horizon', fontsize=13)
    axes[0].set_xticks(range(len(timesteps)))
    axes[0].set_xticklabels(timesteps, rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(len(timesteps)), rmse_values, 'r-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('RMSE vs Prediction Horizon', fontsize=13)
    axes[1].set_xticks(range(len(timesteps)))
    axes[1].set_xticklabels(timesteps, rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(range(len(timesteps)), mae_values, 'g-o', linewidth=2, markersize=8)
    axes[2].set_xlabel('Prediction Horizon', fontsize=12)
    axes[2].set_ylabel('MAE', fontsize=12)
    axes[2].set_title('MAE vs Prediction Horizon', fontsize=13)
    axes[2].set_xticks(range(len(timesteps)))
    axes[2].set_xticklabels(timesteps, rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_per_timestep.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✅ Plots saved to {save_path}')


def evaluate(args):
    """
    Main evaluation function
    """
    # Set device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print(f'Args in experiment: {args}')
    
    # Create experiment
    Exp = Exp_Short_Term_Forecast
    exp = Exp(args)
    
    # Load pretrained model
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f'Loading pretrained model from: {args.checkpoint_path}')
    exp.model.load_state_dict(torch.load(args.checkpoint_path, map_location=exp.device))
    print('✅ Model loaded successfully!')
    
    # Get test data
    test_data, test_loader = exp._get_data(flag='test')
    
    # Evaluation
    print('Starting evaluation on test set...')
    exp.model.eval()
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            
            # Model prediction
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Extract predictions
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
    
    # Concatenate all predictions
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    print(f'Predictions shape: {preds.shape}')
    print(f'Ground truth shape: {trues.shape}')
    
    # Calculate metrics
    print('\nCalculating metrics...')
    metrics = calculate_metrics(preds, trues)
    
    # Print results
    print('\n' + '='*80)
    print('EVALUATION RESULTS')
    print('='*80)
    
    print('\n📊 OVERALL METRICS:')
    print(f"  R² Score:  {metrics['overall']['R2']:.6f}")
    print(f"  RMSE:      {metrics['overall']['RMSE']:.6f}")
    print(f"  MAE:       {metrics['overall']['MAE']:.6f}")
    
    if len(preds.shape) == 3 and preds.shape[2] > 1:
        print('\n📈 PER-FEATURE METRICS:')
        feature_names = ['charge_capacity', 'discharge_capacity', 'internal_resistance', 
                        'temperature_mean', 'temperature_min', 'temperature_max']
        
        for i, feat in enumerate(feature_names[:preds.shape[2]]):
            if feat in metrics:
                print(f"\n  {feat}:")
                print(f"    R² Score:  {metrics[feat]['R2']:.6f}")
                print(f"    RMSE:      {metrics[feat]['RMSE']:.6f}")
                print(f"    MAE:       {metrics[feat]['MAE']:.6f}")
    
    print('\n⏱️  PER-TIMESTEP METRICS:')
    for t, t_metrics in metrics['per_timestep'].items():
        print(f"\n  {t}:")
        print(f"    R² Score:  {t_metrics['R2']:.6f}")
        print(f"    RMSE:      {t_metrics['RMSE']:.6f}")
        print(f"    MAE:       {t_metrics['MAE']:.6f}")
    
    print('\n' + '='*80)
    
    # Save results
    result_path = os.path.join(args.result_path, args.model_id)
    os.makedirs(result_path, exist_ok=True)
    
    # Save metrics to JSON
    import json
    with open(os.path.join(result_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'\n✅ Metrics saved to {os.path.join(result_path, "metrics.json")}')
    
    # Save predictions
    if args.save_predictions:
        save_predictions(preds, trues, os.path.join(result_path, 'predictions'), test_data)
    
    # Create plots
    if args.save_plots:
        plot_results(preds, trues, metrics, os.path.join(result_path, 'plots'), test_data)
    
    print('\n✅ Evaluation completed!')
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Battery Life Prediction Model')
    
    # Basic config
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to pretrained model checkpoint (.pth file)')
    parser.add_argument('--result_path', type=str, default='./evaluation_results/',
                        help='Path to save evaluation results')
    parser.add_argument('--model_id', type=str, default='battery_eval',
                        help='Model identifier for result folder')
    
    # Data config
    parser.add_argument('--root_path', type=str, default='./total/',
                        help='Root path of the data file')
    parser.add_argument('--data', type=str, default='battery', help='Dataset type')
    parser.add_argument('--features', type=str, default='MS',
                        help='Forecasting task: M, S, MS')
    parser.add_argument('--target', type=str, default='discharge_capacity',
                        help='Target feature')
    parser.add_argument('--freq', type=str, default='d', help='Frequency for time features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='Location of model checkpoints (for compatibility)')
    
    # Model config
    parser.add_argument('--model', type=str, default='TimesNet', help='Model name')
    parser.add_argument('--seq_len', type=int, default=24, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=8, help='Start token length')
    parser.add_argument('--pred_len', type=int, default=8, help='Prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=6, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=6, help='Output size')
    parser.add_argument('--d_model', type=int, default=384, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1536, help='Dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='Attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation')
    parser.add_argument('--output_attention', action='store_true', help='Output attention')
    parser.add_argument('--top_k', type=int, default=5, help='TimesNet top_k')
    
    # GPU config
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0', help='Device IDs for multi-GPU')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    # Evaluation config
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader num workers')
    parser.add_argument('--save_predictions', type=bool, default=True, 
                        help='Save predictions to CSV')
    parser.add_argument('--save_plots', type=bool, default=True, 
                        help='Save visualization plots')
    
    # Required for compatibility
    parser.add_argument('--task_name', type=str, default='short_term_forecast')
    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--timeenc', type=int, default=1, help='Time encoding')
    parser.add_argument('--seasonal_patterns', type=str, default=None)
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(args)

