import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    paths = {
        'Mean Teacher (50%)': 'baseline/models/ssl_meanteacher_50/log.csv',
        'CPS (50%)': 'baseline/models/ssl_cps_50/log.csv',
        'UA-MT (50%)': 'baseline/models/ssl_uamt_50/log.csv'
    }

    plt.figure(figsize=(10, 6))
    
    for label, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Find the correct column name for validation IoU
            val_col = [c for c in df.columns if 'val_io_u' in c or 'val_iou' in c][0]
            plt.plot(df['epoch'], df[val_col], label=label, linewidth=2)
    
    plt.title('Validation IoU Comparison: SSL at 50% Labeled Data', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation IoU', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs('baseline/logs/verification/plots', exist_ok=True)
    out_path = 'baseline/logs/verification/plots/ssl_50_iou_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
