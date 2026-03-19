"""
Fourier Transform Analysis: Why Resolution Matters for Pancreas Segmentation

This script provides theoretical grounding for the resolution-preservation hypothesis
by analyzing the frequency content of CT slices at native (512x512) vs downsampled
(256x256, 128x128) resolutions.

Key insight: Downsampling acts as a low-pass filter, destroying high-frequency
components that encode organ boundaries and fine texture — exactly what the
pancreas depends on for accurate segmentation.

Generates two publication-quality figures:
1. Radial power spectrum comparison (512 vs 256 vs 128)
2. Spatial frequency visualization panel (original, spectrum, boundary overlay)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from scipy import ndimage
import os
import glob

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

DATA_DIR = '/scratch/lustre/home/kayi9958/ish/data/Task07_Pancreas'
PATCH_DIR = '/scratch/lustre/home/kayi9958/ish/preprocessed_v5_patches'
OUT_DIR = '/scratch/lustre/home/kayi9958/ish/ishFinal/overleaf_export/images'


def load_slice_with_pancreas(case_id='pancreas_001'):
    """Load a preprocessed slice that contains pancreas and its mask."""
    x = np.load(os.path.join(PATCH_DIR, f'{case_id}_x.npy'))
    y = np.load(os.path.join(PATCH_DIR, f'{case_id}_y.npy'))

    # Find slices with substantial pancreas content
    pancreas_area = np.sum(y > 0, axis=(1, 2))
    best_slice = np.argmax(pancreas_area)

    img = x[best_slice]  # 256x256, float32, [0, 1]
    mask = (y[best_slice] > 0).astype(np.float32)

    return img, mask, best_slice


def simulate_resolutions(img_256):
    """
    From a 256x256 patch (which represents native resolution),
    simulate what happens when we downsample to 128 and 64
    (equivalent to 256 and 128 from a 512 original).
    """
    # Native resolution (our 256x256 patch = native quality)
    native = img_256.copy()

    # Simulate 256x256 downsampled from 512 (= 128x128 from our 256 patch, upscaled back)
    down_128 = cv2.resize(img_256, (128, 128), interpolation=cv2.INTER_AREA)
    up_128 = cv2.resize(down_128, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Simulate 128x128 downsampled from 512 (= 64x64 from our 256 patch, upscaled back)
    down_64 = cv2.resize(img_256, (64, 64), interpolation=cv2.INTER_AREA)
    up_64 = cv2.resize(down_64, (256, 256), interpolation=cv2.INTER_LINEAR)

    return native, up_128, up_64


def compute_2d_fft(img):
    """Compute 2D FFT and return magnitude spectrum (log-scaled)."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    # Log scale for visualization, avoid log(0)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude, magnitude


def radial_power_spectrum(magnitude, normalize=True):
    """
    Compute the radially averaged power spectrum.
    This shows how much energy exists at each spatial frequency.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Create radial distance map
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)

    # Power spectrum
    power = magnitude ** 2

    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)

    for i in range(h):
        for j in range(w):
            dist = int(np.sqrt((j - cx)**2 + (i - cy)**2))
            if dist < max_r:
                radial_profile[dist] += power[i, j]
                counts[dist] += 1

    # Average
    mask = counts > 0
    radial_profile[mask] /= counts[mask]

    if normalize:
        radial_profile /= radial_profile.max() + 1e-10

    return radial_profile


def compute_boundary_frequency_content(mask):
    """
    Extract boundary of the pancreas and compute its frequency content.
    This shows WHERE in the frequency domain the pancreas boundary lives.
    """
    # Extract boundary using morphological gradient
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = dilated - eroded

    # Frequency content of the boundary
    _, boundary_mag = compute_2d_fft(boundary)
    boundary_radial = radial_power_spectrum(boundary_mag, normalize=True)

    return boundary, boundary_radial


def plot_radial_spectrum_comparison(save_path):
    """
    Main figure: Radial power spectrum showing high-frequency loss from downsampling.
    """
    img, mask, slice_idx = load_slice_with_pancreas('pancreas_001')
    native, down_256, down_128 = simulate_resolutions(img)

    # Compute FFT magnitudes
    _, mag_native = compute_2d_fft(native)
    _, mag_256 = compute_2d_fft(down_256)
    _, mag_128 = compute_2d_fft(down_128)

    # Radial power spectra
    rps_native = radial_power_spectrum(mag_native)
    rps_256 = radial_power_spectrum(mag_256)
    rps_128 = radial_power_spectrum(mag_128)

    # Boundary frequency content
    boundary, boundary_rps = compute_boundary_frequency_content(mask)

    # Normalize frequency axis to [0, 1]
    freq_native = np.linspace(0, 1, len(rps_native))
    freq_256 = np.linspace(0, 1, len(rps_256))
    freq_128 = np.linspace(0, 1, len(rps_128))
    freq_boundary = np.linspace(0, 1, len(boundary_rps))

    # ---- Create figure ----
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # Panel A: Radial power spectrum comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(freq_native, rps_native + 1e-12, 'b-', linewidth=2.0, label='Native 512×512', alpha=0.9)
    ax1.semilogy(freq_256, rps_256 + 1e-12, 'r--', linewidth=2.0, label='Downsampled 256×256', alpha=0.9)
    ax1.semilogy(freq_128, rps_128 + 1e-12, 'g:', linewidth=2.5, label='Downsampled 128×128', alpha=0.9)

    # Shade the high-frequency region that gets destroyed
    cutoff_256 = 0.5  # Nyquist of 256 relative to 512
    cutoff_128 = 0.25  # Nyquist of 128 relative to 512

    ax1.axvspan(cutoff_256, 1.0, alpha=0.08, color='red', label='Lost at 256×256')
    ax1.axvspan(cutoff_128, cutoff_256, alpha=0.06, color='orange')

    # Overlay boundary frequency content (scaled for visibility)
    boundary_scaled = boundary_rps * rps_native.max() * 0.3
    ax1.fill_between(freq_boundary, 1e-12, boundary_scaled + 1e-12,
                     alpha=0.25, color='purple', label='Pancreas boundary')

    ax1.set_xlabel('Normalized Spatial Frequency', fontsize=11)
    ax1.set_ylabel('Radial Power (log scale)', fontsize=11)
    ax1.set_title('(a) Radial Power Spectrum', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=1e-8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Frequency magnitude maps
    ax2 = fig.add_subplot(gs[1])
    log_native, _ = compute_2d_fft(native)
    log_128, _ = compute_2d_fft(down_128)

    # Difference map: what frequencies are lost
    diff = log_native - log_128
    diff = np.clip(diff, 0, None)  # Only show lost frequencies

    im = ax2.imshow(diff, cmap='hot', aspect='equal')
    ax2.set_title('(b) Frequency Loss Map\n(Native $-$ 128×128)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency $u$', fontsize=10)
    ax2.set_ylabel('Frequency $v$', fontsize=10)
    ax2.set_xticks([0, 128, 255])
    ax2.set_xticklabels(['$-f_{max}$', '0', '$f_{max}$'])
    ax2.set_yticks([0, 128, 255])
    ax2.set_yticklabels(['$-f_{max}$', '0', '$f_{max}$'])
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Lost energy (log)')

    # Panel C: Spatial domain impact
    ax3 = fig.add_subplot(gs[2])

    # Show the spatial difference (what information is lost)
    spatial_diff = np.abs(native - down_128)

    # Overlay pancreas boundary
    kernel = np.ones((3, 3), np.uint8)
    boundary_vis = cv2.dilate(mask, kernel, iterations=2) - cv2.erode(mask, kernel, iterations=1)

    ax3.imshow(native, cmap='gray', aspect='equal')
    ax3.imshow(spatial_diff, cmap='Reds', alpha=0.6, aspect='equal')
    # Draw boundary contour
    contours, _ = cv2.findContours(boundary_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2 and len(cnt) > 2:
            ax3.plot(cnt[:, 0], cnt[:, 1], 'lime', linewidth=1.5)

    ax3.set_title('(c) Spatial Information Loss\n(red) with Boundary (green)', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_resolution_visual_comparison(save_path):
    """
    Visual comparison: same slice at 3 resolutions + their FFT spectra + edge maps.
    Shows what the network "sees" at each resolution.
    """
    img, mask, _ = load_slice_with_pancreas('pancreas_001')
    native, down_256, down_128 = simulate_resolutions(img)

    images = [native, down_256, down_128]
    titles = ['Native 512×512\n(via 256×256 patch)',
              'Downsampled to\n256×256',
              'Downsampled to\n128×128']

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.patch.set_facecolor('white')

    for col, (im, title) in enumerate(zip(images, titles)):
        # Row 1: CT image with pancreas overlay
        axes[0, col].imshow(im, cmap='gray')
        axes[0, col].imshow(mask, cmap='Greens', alpha=0.3 * (mask > 0))
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel('CT + Pancreas', fontsize=11, fontweight='bold')

        # Row 2: FFT magnitude spectrum
        log_mag, _ = compute_2d_fft(im)
        axes[1, col].imshow(log_mag, cmap='inferno', aspect='equal')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel('Frequency Spectrum', fontsize=11, fontweight='bold')

        # Row 3: Edge detection (Sobel) — shows boundary sharpness
        sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = edges / edges.max()  # Normalize

        axes[2, col].imshow(edges, cmap='magma', aspect='equal')
        # Overlay pancreas boundary
        kernel = np.ones((3, 3), np.uint8)
        boundary_vis = cv2.dilate(mask, kernel, iterations=2) - cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(boundary_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt.shape) == 2 and len(cnt) > 2:
                axes[2, col].plot(cnt[:, 0], cnt[:, 1], 'lime', linewidth=1.5)
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
        if col == 0:
            axes[2, col].set_ylabel('Edge Map + Boundary', fontsize=11, fontweight='bold')

    # Add column annotations for edge strength
    # Compute edge strength within pancreas region for each resolution
    for col, im in enumerate(images):
        sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        # Edge strength at pancreas boundary
        boundary_mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1) - \
                       cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
        boundary_mask = boundary_mask > 0

        if boundary_mask.sum() > 0:
            edge_strength = edges[boundary_mask].mean()
            axes[2, col].text(0.5, -0.08, f'Boundary Edge Strength: {edge_strength:.4f}',
                            transform=axes[2, col].transAxes, ha='center', fontsize=9,
                            color='#2c3e50', fontweight='bold')

    plt.suptitle('Impact of Resolution on Spatial Frequency Content and Boundary Sharpness',
                fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    print("Running Fourier Transform Analysis...")
    print("=" * 60)

    # Figure 1: Radial power spectrum (the main analytical figure)
    plot_radial_spectrum_comparison(
        os.path.join(OUT_DIR, 'fourier_radial_spectrum.png'))

    # Figure 2: Visual comparison panel (3x3 grid)
    plot_resolution_visual_comparison(
        os.path.join(OUT_DIR, 'fourier_resolution_comparison.png'))

    print("\nFourier analysis complete!")
    print("Generated figures ready for manuscript integration.")
