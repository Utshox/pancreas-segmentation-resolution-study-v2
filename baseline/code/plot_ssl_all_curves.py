import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_regime(ax, paths, title, max_epoch=None):
    for label, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if max_epoch:
                df = df[df['epoch'] <= max_epoch]
            # Find the correct column name for validation IoU
            val_cols = [c for c in df.columns if 'val_io_u' in c or 'val_iou' in c]
            if val_cols:
                val_col = val_cols[0]
                ax.plot(df['epoch'], df[val_col], label=label, linewidth=2)
    
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation IoU', fontsize=14)
    if max_epoch:
        ax.set_xlim(0, max_epoch)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Validation IoU Comparison Across SSL Data Regimes', fontsize=20, fontweight='bold', y=1.05)
    
    paths_10 = {
        'Mean Teacher': 'baseline/models/ssl_meanteacher_10/log.csv',
        'CPS': 'baseline/models/ssl_cps_10/log.csv',
        'UA-MT': 'baseline/models/ssl_uamt_10/log.csv'
    }
    plot_regime(axes[0], paths_10, '10% Labeled Data', max_epoch=100)
    
    paths_25 = {
        'Mean Teacher': 'baseline/models/ssl_meanteacher_25/log.csv',
        'CPS': 'baseline/models/ssl_cps_25/log.csv',
        'UA-MT': 'baseline/models/ssl_uamt_25/log.csv'
    }
    plot_regime(axes[1], paths_25, '25% Labeled Data', max_epoch=100)
    
    paths_50 = {
        'Mean Teacher': 'baseline/models/ssl_meanteacher_50/log.csv',
        'CPS': 'baseline/models/ssl_cps_50/log.csv',
        'UA-MT': 'baseline/models/ssl_uamt_50/log.csv'
    }
    plot_regime(axes[2], paths_50, '50% Labeled Data', max_epoch=68)
    
    plt.tight_layout()
    os.makedirs('baseline/logs/verification/plots', exist_ok=True)
    out_path = 'baseline/logs/verification/plots/ssl_all_iou_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
