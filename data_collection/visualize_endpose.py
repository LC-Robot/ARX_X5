#!/usr/bin/env python3
"""
Visualize end-effector trajectory (position & orientation) from HDF5.
Plots XYZ position and RPY orientation curves, with simple jitter analysis.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def jitter_analysis(data, labels, thresh_jump=0.02, high_freq_ratio=50):
    """
    Simple jitter diagnostics on each dimension.
    - high-frequency oscillation: sign changes of diff over samples
    - large jumps: max absolute diff
    """
    print("\n[INFO] Jitter analysis:")
    for i, name in enumerate(labels):
        diff = np.diff(data[:, i])
        if len(diff) == 0:
            continue
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        vibration_ratio = sign_changes / len(diff) * 100
        max_jump = np.max(np.abs(diff))

        issues = []
        if vibration_ratio > high_freq_ratio:
            issues.append(f"high-frequency ({vibration_ratio:.1f}% sign changes)")
        if max_jump > thresh_jump:
            issues.append(f"large jump {max_jump:.4f}")

        if issues:
            print(f"  ⚠️  {name}: " + "; ".join(issues))
        else:
            print(f"  {name}: OK (max jump {max_jump:.4f}, vibration {vibration_ratio:.1f}%)")
    print()


def plot_endpose(hdf5_path, save_path=None, show=True):
    """
    Plot end-effector position (XYZ) and orientation (roll/pitch/yaw) over time.
    """
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] File not found: {hdf5_path}")
        return

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "state/end_effector/position" not in f or "state/end_effector/euler" not in f:
                print("[ERROR] Missing end_effector position or euler in file.")
                return
            pos = f["state/end_effector/position"][:]  # [N,3]
            euler = f["state/end_effector/euler"][:]    # [N,3]
            timestamps = f["observation/rgb_timestamp"][:] if "observation/rgb_timestamp" in f else None
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return

    n_steps = pos.shape[0]
    print(f"[INFO] Loaded end-effector data: steps={n_steps}")

    # time axis
    if timestamps is not None and len(timestamps) == n_steps:
        t_axis = timestamps - timestamps[0]
        t_label = "Time (s)"
        print(f"[INFO] Time span: {t_axis[0]:.2f} - {t_axis[-1]:.2f} s")
    else:
        t_axis = np.arange(n_steps)
        t_label = "Steps"
        print(f"[INFO] Step range: 0 - {n_steps}")

    # jitter diagnostics
    jitter_analysis(pos, ["X", "Y", "Z"], thresh_jump=0.01, high_freq_ratio=50)
    jitter_analysis(euler, ["Roll", "Pitch", "Yaw"], thresh_jump=0.02, high_freq_ratio=50)

    # downsample for display if too long
    def downsample(arr, t):
        if arr.shape[0] > 5000:
            step = max(1, arr.shape[0] // 2000)
            idx = np.arange(0, arr.shape[0], step)
            return arr[idx], t[idx], idx
        return arr, t, None

    pos_plot, t_pos, _ = downsample(pos, t_axis)
    euler_plot, t_eu, _ = downsample(euler, t_axis)

    fig = plt.figure(figsize=(14, 8))
    ep_name = os.path.basename(os.path.dirname(hdf5_path))
    fig.suptitle(f'End-effector Trajectory - {ep_name}', fontsize=14, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    labels_pos = ["X", "Y", "Z"]
    labels_eu = ["Roll", "Pitch", "Yaw"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

    # position plots
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t_pos, pos_plot[:, i], color=colors[i], linewidth=1.2)
        ax.set_xlabel(t_label)
        ax.set_ylabel(f'{labels_pos[i]} (m)')
        ax.set_title(f'{labels_pos[i]} position', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        stats = f"Min: {np.min(pos[:, i]):+.4f}\nMax: {np.max(pos[:, i]):+.4f}\nStd: {np.std(pos[:, i]):.4f}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=7, va='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # orientation plots
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t_eu, euler_plot[:, i], color=colors[i+3], linewidth=1.2)
        ax.set_xlabel(t_label)
        ax.set_ylabel(f'{labels_eu[i]} (rad)')
        ax.set_title(f'{labels_eu[i]} (RPY)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        stats = f"Min: {np.min(euler[:, i]):+.4f}\nMax: {np.max(euler[:, i]):+.4f}\nStd: {np.std(euler[:, i]):.4f}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=7, va='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize end-effector trajectory")
    parser.add_argument("hdf5_path", type=str, help="HDF5 file path")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure")
    parser.add_argument("--no_show", action="store_true", help="Do not show window")
    args = parser.parse_args()

    print("=" * 70)
    print("End-effector Trajectory Visualization")
    print("=" * 70)
    print(f"File: {args.hdf5_path}")
    print("=" * 70)
    print()

    plot_endpose(
        args.hdf5_path,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

