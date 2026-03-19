import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_comparison(log_256, log_128, log_transformer, output_dir):
    """
    Reads three Keras log.csv files and plots a comparison.
    """
    if not os.path.exists(log_256) or not os.path.exists(log_128):
        print("Error: One or both ablation log files not found.")
        return

    df_256 = pd.read_csv(log_256)
    df_128 = pd.read_csv(log_128)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # Plot: Validation IoU Comparison
    if 'val_io_u' in df_256.columns and 'val_io_u' in df_128.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_256['epoch'] + 1, df_256['val_io_u'], label='256x256 Full-Slice', color='blue', linewidth=2.5)
        plt.plot(df_128['epoch'] + 1, df_128['val_io_u'], label='128x128 Full-Slice', color='orange', linewidth=2.5)
        
        if log_transformer and os.path.exists(log_transformer):
            df_transformer = pd.read_csv(log_transformer)
            if 'val_io_u' in df_transformer.columns:
                plt.plot(df_transformer['epoch'] + 1, df_transformer['val_io_u'], label='Transformer Baseline', color='green', linewidth=2.5, linestyle='--')

        plt.title('Validation IoU: Resolution Ablation & Transformer', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Intersection over Union (IoU)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_iou_comparison.png'), dpi=300)
        plt.close()
        print(f"Comparison plot saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot comparison of ablation logs")
    parser.add_argument('--log_256', type=str, required=True, help="Path to 256x256 log.csv")
    parser.add_argument('--log_128', type=str, required=True, help="Path to 128x128 log.csv")
    parser.add_argument('--log_transformer', type=str, required=False, help="Path to transformer log.csv")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    
    plot_comparison(args.log_256, args.log_128, args.log_transformer, args.output_dir)

if __name__ == "__main__":
    main()
