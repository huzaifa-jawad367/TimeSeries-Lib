"""
Example script showing how to use the BatteryDataset class with TimesNet
"""

import argparse
import torch
from data_provider.data_factory import data_provider

def create_args():
    """Create example arguments for battery dataset"""
    args = argparse.Namespace()
    
    # Dataset configuration
    args.data = 'battery'
    args.root_path = './total'  # Directory containing CSV files
    args.data_path = None  # Will use b1c0 for train/val, other files for test
    
    # Task configuration (REQUIRED for data_provider)
    args.task_name = 'short_term_forecast'  # This is required!
    
    # Model configuration
    args.seq_len = 24      # Context window (cycles)
    args.label_len = 8     # Label length 
    args.pred_len = 8      # Prediction horizon (cycles)
    
    # Features and target
    args.features = 'MS'   # Multi-variate time series
    args.target = 'discharge_capacity'
    
    # Training configuration
    args.batch_size = 32
    args.num_workers = 4
    args.embed = 'timeF'   # Time feature encoding
    
    # Data processing
    args.scale = True
    args.freq = 'D'        # Daily frequency (cycle-based)
    args.seasonal_patterns = None
    
    # Augmentation (optional)
    args.augmentation_ratio = 0.0
    
    return args

def main():
    """Example usage of battery dataset"""
    args = create_args()
    
    # Update the path to your actual battery data directory
    args.root_path = './total'
    
    print("Creating battery datasets...")
    
    # Create train, validation, and test datasets
    train_dataset, train_loader = data_provider(args, flag='train')
    val_dataset, val_loader = data_provider(args, flag='val')
    test_dataset, test_loader = data_provider(args, flag='test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get a sample batch
    print("\nSample batch:")
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        print(f"Input shape: {batch_x.shape}")      # (batch_size, seq_len, num_features)
        print(f"Target shape: {batch_y.shape}")     # (batch_size, label_len + pred_len, num_features)
        print(f"Input time features shape: {batch_x_mark.shape}")
        print(f"Target time features shape: {batch_y_mark.shape}")
        print(f"Feature names: {train_dataset.feature_names}")
        break
    
    print("\nDataset ready for TimesNet training!")

if __name__ == "__main__":
    main()
