"""
Generate publication-quality diagrams for the journal manuscript.
1. U-Net Architecture Diagram
2. SSL Framework Overview (MT / UA-MT / CPS pipeline)
3. Methodology Overview (full pipeline)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# Style settings for publication
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# Color palette
ENCODER_COLOR = '#3498db'   # blue
DECODER_COLOR = '#e74c3c'   # red
BOTTLENECK_COLOR = '#2c3e50'  # dark
SKIP_COLOR = '#27ae60'      # green
POOL_COLOR = '#95a5a6'      # gray
OUTPUT_COLOR = '#f39c12'    # orange
BG_COLOR = '#fafafa'

def draw_unet_architecture(save_path):
    """Draw a clean U-Net architecture schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Encoder blocks (left side, going down)
    enc_configs = [
        # (x, y, width, height, channels, size)
        (0.5, 6.0, 1.2, 1.2, '32', '256²'),
        (2.0, 4.5, 1.2, 1.0, '64', '128²'),
        (3.5, 3.2, 1.2, 0.8, '128', '64²'),
        (5.0, 2.1, 1.2, 0.6, '256', '32²'),
    ]

    # Bottleneck
    bottle = (6.5, 1.2, 1.2, 0.5, '512', '16²')

    # Decoder blocks (right side, going up)
    dec_configs = [
        (8.0, 2.1, 1.2, 0.6, '256', '32²'),
        (9.5, 3.2, 1.2, 0.8, '128', '64²'),
        (11.0, 4.5, 1.2, 1.0, '64', '128²'),
        (12.5, 6.0, 1.2, 1.2, '32', '256²'),
    ]

    def draw_block(x, y, w, h, color, label_top, label_bot, alpha=0.85):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h + 0.12, label_top, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#2c3e50')
        ax.text(x + w/2, y - 0.12, label_bot, ha='center', va='top',
                fontsize=7, color='#7f8c8d')

    # Draw encoder
    for (x, y, w, h, ch, sz) in enc_configs:
        draw_block(x, y, w, h, ENCODER_COLOR, ch, sz)

    # Draw bottleneck
    bx, by, bw, bh, bch, bsz = bottle
    draw_block(bx, by, bw, bh, BOTTLENECK_COLOR, bch, bsz, alpha=0.95)

    # Draw decoder
    for (x, y, w, h, ch, sz) in dec_configs:
        draw_block(x, y, w, h, DECODER_COLOR, ch, sz)

    # Encoder downsampling arrows
    for i in range(len(enc_configs) - 1):
        x1 = enc_configs[i][0] + enc_configs[i][2]
        y1 = enc_configs[i][1] + enc_configs[i][3] / 2
        x2 = enc_configs[i+1][0]
        y2 = enc_configs[i+1][1] + enc_configs[i+1][3] / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=POOL_COLOR, lw=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, 'Pool', fontsize=6, ha='center', color=POOL_COLOR)

    # Last encoder to bottleneck
    x1 = enc_configs[-1][0] + enc_configs[-1][2]
    y1 = enc_configs[-1][1] + enc_configs[-1][3] / 2
    x2 = bottle[0]
    y2 = bottle[1] + bottle[3] / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=POOL_COLOR, lw=1.5))

    # Bottleneck to first decoder
    x1 = bottle[0] + bottle[2]
    y1 = bottle[1] + bottle[3] / 2
    x2 = dec_configs[0][0]
    y2 = dec_configs[0][1] + dec_configs[0][3] / 2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my + 0.2, 'Up', fontsize=6, ha='center', color='#e74c3c')

    # Decoder upsampling arrows
    for i in range(len(dec_configs) - 1):
        x1 = dec_configs[i][0] + dec_configs[i][2]
        y1 = dec_configs[i][1] + dec_configs[i][3] / 2
        x2 = dec_configs[i+1][0]
        y2 = dec_configs[i+1][1] + dec_configs[i+1][3] / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, 'Up', fontsize=6, ha='center', color='#e74c3c')

    # Skip connections (curved arrows from encoder to decoder)
    skip_pairs = list(zip(enc_configs, reversed(dec_configs)))
    for (ex, ey, ew, eh, _, _), (dx, dy, dw, dh, _, _) in skip_pairs:
        ex_mid = ex + ew / 2
        ey_top = ey + eh
        dx_mid = dx + dw / 2
        dy_top = dy + dh
        ax.annotate('', xy=(dx_mid, dy_top), xytext=(ex_mid, ey_top),
                    arrowprops=dict(arrowstyle='->', color=SKIP_COLOR, lw=1.5,
                                   connectionstyle='arc3,rad=-0.15', linestyle='--'))

    # Output block
    ox, oy = 14.0, 6.3
    rect = FancyBboxPatch((ox, oy), 0.6, 0.6, boxstyle="round,pad=0.05",
                          facecolor=OUTPUT_COLOR, edgecolor='black', linewidth=1.0, alpha=0.9)
    ax.add_patch(rect)
    ax.text(ox + 0.3, oy + 0.72, '1×1 Conv', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.text(ox + 0.3, oy - 0.12, 'Sigmoid', ha='center', va='top', fontsize=7, color='#7f8c8d')

    # Arrow from last decoder to output
    ax.annotate('', xy=(ox, oy + 0.3), xytext=(dec_configs[-1][0] + dec_configs[-1][2], dec_configs[-1][1] + dec_configs[-1][3]/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Input label
    ax.text(0.5 + 0.6, 7.5, 'Input\n256×256×1', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    # Output label
    ax.text(ox + 0.3, oy - 0.5, 'Output\n256×256×1', ha='center', va='top',
            fontsize=9, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ENCODER_COLOR, edgecolor='black', label='Encoder (Conv + ReLU)'),
        mpatches.Patch(facecolor=DECODER_COLOR, edgecolor='black', label='Decoder (TransConv + ReLU)'),
        mpatches.Patch(facecolor=BOTTLENECK_COLOR, edgecolor='black', label='Bottleneck'),
        plt.Line2D([0], [0], color=SKIP_COLOR, linestyle='--', lw=1.5, label='Skip Connection'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=7,
              frameon=True, fancybox=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 8.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def draw_ssl_framework(save_path):
    """Draw the SSL framework showing MT, UA-MT, and CPS pipelines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    titles = ['(a) Mean Teacher', '(b) UA-MT (Ours)', '(c) Cross-Pseudo Supervision']

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        if idx == 0:  # Mean Teacher
            # Student
            rect = FancyBboxPatch((1, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#3498db', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(2.5, 6.75, 'Student\n$f_\\theta$', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # Teacher
            rect = FancyBboxPatch((6, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#e74c3c', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(7.5, 6.75, 'Teacher\n$f_{\\theta\'}$ (EMA)', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # EMA arrow
            ax.annotate('', xy=(6, 7.5), xytext=(4, 7.5),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
            ax.text(5, 7.8, 'EMA\n$\\alpha=0.999$', ha='center', fontsize=7, color='#2c3e50')

            # Labeled data
            rect = FancyBboxPatch((0.5, 3), 2.5, 1, boxstyle="round,pad=0.1",
                                  facecolor='#27ae60', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(1.75, 3.5, 'Labeled $x_l$', ha='center', va='center', fontsize=8, color='white')

            # Unlabeled data
            rect = FancyBboxPatch((3.5, 3), 2.5, 1, boxstyle="round,pad=0.1",
                                  facecolor='#f39c12', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(4.75, 3.5, 'Unlabeled $x_u$', ha='center', va='center', fontsize=8, color='white')

            # Arrows
            ax.annotate('', xy=(2.5, 6), xytext=(1.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
            ax.annotate('', xy=(2.5, 6), xytext=(4.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))
            ax.annotate('', xy=(7.5, 6), xytext=(4.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

            # Loss
            rect = FancyBboxPatch((2.5, 0.5), 5, 1.2, boxstyle="round,pad=0.1",
                                  facecolor='#ecf0f1', edgecolor='#2c3e50', alpha=0.9)
            ax.add_patch(rect)
            ax.text(5, 1.1, '$\\mathcal{L} = \\mathcal{L}_{sup} + \\lambda \\cdot MSE$',
                    ha='center', va='center', fontsize=9, color='#2c3e50')

            ax.annotate('', xy=(5, 1.7), xytext=(2.5, 6),
                        arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1, linestyle='--'))
            ax.annotate('', xy=(5, 1.7), xytext=(7.5, 6),
                        arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1, linestyle='--'))

        elif idx == 1:  # UA-MT
            # Student
            rect = FancyBboxPatch((1, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#3498db', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(2.5, 6.75, 'Student\n$f_\\theta$', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # Teacher with MC Dropout
            rect = FancyBboxPatch((6, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#e74c3c', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(7.5, 6.75, 'Teacher\n+ MC Dropout', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # EMA arrow
            ax.annotate('', xy=(6, 7.5), xytext=(4, 7.5),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
            ax.text(5, 7.8, 'EMA', ha='center', fontsize=7, color='#2c3e50')

            # Uncertainty block
            rect = FancyBboxPatch((6.2, 4.2), 2.6, 1, boxstyle="round,pad=0.1",
                                  facecolor='#9b59b6', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(7.5, 4.7, 'Uncertainty\n$T=8$ passes', ha='center', va='center', fontsize=8, color='white')

            ax.annotate('', xy=(7.5, 5.2), xytext=(7.5, 6),
                        arrowprops=dict(arrowstyle='<-', color='#9b59b6', lw=1.5))

            # Data
            rect = FancyBboxPatch((0.5, 3), 2.5, 1, boxstyle="round,pad=0.1",
                                  facecolor='#27ae60', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(1.75, 3.5, 'Labeled $x_l$', ha='center', va='center', fontsize=8, color='white')

            rect = FancyBboxPatch((3.5, 3), 2.5, 1, boxstyle="round,pad=0.1",
                                  facecolor='#f39c12', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(4.75, 3.5, 'Unlabeled $x_u$', ha='center', va='center', fontsize=8, color='white')

            ax.annotate('', xy=(2.5, 6), xytext=(1.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
            ax.annotate('', xy=(2.5, 6), xytext=(4.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))
            ax.annotate('', xy=(7.5, 6), xytext=(4.75, 4),
                        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

            # Loss with uncertainty weighting
            rect = FancyBboxPatch((1.5, 0.5), 7, 1.2, boxstyle="round,pad=0.1",
                                  facecolor='#ecf0f1', edgecolor='#2c3e50', alpha=0.9)
            ax.add_patch(rect)
            ax.text(5, 1.1, '$\\mathcal{L} = \\mathcal{L}_{sup} + \\lambda \\cdot (1-u) \\cdot MSE$',
                    ha='center', va='center', fontsize=9, color='#2c3e50')

            ax.annotate('', xy=(5, 1.7), xytext=(7.5, 4.2),
                        arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1, linestyle='--'))

        elif idx == 2:  # CPS
            # Network A
            rect = FancyBboxPatch((1, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#3498db', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(2.5, 6.75, 'Network A\n$f_A$', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # Network B
            rect = FancyBboxPatch((6, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#e74c3c', edgecolor='black', alpha=0.85)
            ax.add_patch(rect)
            ax.text(7.5, 6.75, 'Network B\n$f_B$', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

            # Cross arrows
            ax.annotate('', xy=(6, 7.2), xytext=(4, 7.2),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
            ax.annotate('', xy=(4, 6.3), xytext=(6, 6.3),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
            ax.text(5, 7.7, 'Pseudo-labels', ha='center', fontsize=7, color='#2c3e50')
            ax.text(5, 5.9, 'Pseudo-labels', ha='center', fontsize=7, color='#2c3e50')

            # Data
            rect = FancyBboxPatch((3, 3), 4, 1, boxstyle="round,pad=0.1",
                                  facecolor='#27ae60', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(5, 3.5, 'Labeled + Unlabeled', ha='center', va='center', fontsize=8, color='white')

            ax.annotate('', xy=(2.5, 6), xytext=(4, 4),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
            ax.annotate('', xy=(7.5, 6), xytext=(6, 4),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

            # Loss
            rect = FancyBboxPatch((1.5, 0.5), 7, 1.2, boxstyle="round,pad=0.1",
                                  facecolor='#ecf0f1', edgecolor='#2c3e50', alpha=0.9)
            ax.add_patch(rect)
            ax.text(5, 1.1, '$\\mathcal{L} = \\mathcal{L}_{sup}^{A+B} + \\lambda \\cdot \\mathcal{L}_{cross}$',
                    ha='center', va='center', fontsize=9, color='#2c3e50')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def draw_pipeline_overview(save_path):
    """Draw the complete methodology pipeline overview."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    steps = [
        (0.3, 1.5, 2.2, 2.0, '#3498db', 'CT Volume\n512×512\nNative Resolution', 'Input'),
        (3.0, 1.5, 2.2, 2.0, '#27ae60', 'HU Windowing\n[-125, 275]\n+ Normalize', 'Preprocessing'),
        (5.7, 1.5, 2.2, 2.0, '#f39c12', 'Patch Extraction\n256×256\nRandom Crops', 'Resolution\nPreservation'),
        (8.4, 1.5, 2.2, 2.0, '#e74c3c', '2D U-Net\n7.8M Params\n(or SSL)', 'Training'),
        (11.1, 1.5, 2.2, 2.0, '#9b59b6', 'Sliding Window\nStride=128\nGaussian Blend', 'Inference'),
        (13.8, 1.5, 1.8, 2.0, '#2c3e50', '3D Volume\nReconstruction\n0.849 Dice', 'Output'),
    ]

    for (x, y, w, h, color, text, label) in steps:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', linespacing=1.3)
        ax.text(x + w/2, y - 0.3, label, ha='center', va='top',
                fontsize=8, color='#2c3e50', fontstyle='italic')

    # Arrows between steps
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + steps[i][2]
        y1 = steps[i][1] + steps[i][3] / 2
        x2 = steps[i+1][0]
        y2 = steps[i+1][1] + steps[i+1][3] / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    ax.set_title('High-Resolution Patch-Based Framework for Pancreas Segmentation',
                 fontsize=13, fontweight='bold', color='#2c3e50', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    out_dir = '/scratch/lustre/home/kayi9958/ish/ishFinal/overleaf_export/images'

    draw_unet_architecture(f'{out_dir}/architecture_unet.png')
    draw_ssl_framework(f'{out_dir}/ssl_framework_overview.png')
    draw_pipeline_overview(f'{out_dir}/pipeline_overview.png')

    print("\nAll diagrams generated successfully!")
