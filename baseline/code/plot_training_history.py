import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_history(log_path, output_dir, exp_name):
    """
    Reads a Keras log.csv and generates publication-ready plots.
    """
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    df = pd.read_csv(log_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # Plot 1: Loss
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'] + 1, df['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(df['epoch'] + 1, df['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
    plt.title(f'Training & Validation Loss ({exp_name})', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{exp_name}_loss_curve.png'), dpi=300)
    plt.close()

    # Plot 2: IoU Metric
    if 'val_io_u' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'] + 1, df['io_u'], label='Training IoU', color='blue', linewidth=2)
        plt.plot(df['epoch'] + 1, df['val_io_u'], label='Validation IoU', color='green', linewidth=2, linestyle='--')
        plt.title(f'Training & Validation IoU ({exp_name})', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Intersection over Union (IoU)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name}_iou_curve.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training history from log.csv")
    parser.add_argument('--log_csv', type=str, required=True, help="Path to log.csv")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save plots")
    parser.add_argument('--exp_name', type=str, default="Experiment", help="Name of the experiment for titles")
    args = parser.parse_args()
    
    plot_history(args.log_csv, args.output_dir, args.exp_name)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
