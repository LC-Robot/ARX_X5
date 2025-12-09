#!/usr/bin/env python3
"""
Trajectory visualization tool

Reads state data from an HDF5 file and plots 6 joint trajectories plus gripper width.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_trajectory(hdf5_path, save_path=None, show=True):
    """
    Plot joint trajectories

    Args:
        hdf5_path: path to HDF5 file
        save_path: optional path to save figure
        show: whether to show window
    """
    
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] File not found: {hdf5_path}")
        return
    
    # read data
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'state/joint/position' not in f:
                print(f"[ERROR] Missing 'state/joint/position'")
                return
            
            joint_pos = f['state/joint/position'][:]
            
            if 'state/joint/gripper_width' not in f:
                print(f"[ERROR] Missing 'state/joint/gripper_width'")
                return
            
            gripper_width = f['state/joint/gripper_width'][:]
            
            timestamps = None
            if 'observation/rgb_timestamp' in f:
                timestamps = f['observation/rgb_timestamp'][:]
    
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return
    
    n_steps = joint_pos.shape[0]
    n_joints = joint_pos.shape[1]
    
    print(f"[INFO] Data loaded:")
    print(f"  Steps: {n_steps}")
    print(f"  Joints: {n_joints}")
    
    print(f"\n[INFO] Joint ranges:")
    for i in range(n_joints):
        min_val = np.min(joint_pos[:, i])
        max_val = np.max(joint_pos[:, i])
        std_val = np.std(joint_pos[:, i])
        print(f"  Joint {i+1}: [{min_val:+.4f}, {max_val:+.4f}], std={std_val:.4f}")
    
    print(f"  Gripper: [{np.min(gripper_width):.4f}, {np.max(gripper_width):.4f}], std={np.std(gripper_width):.4f}")
    
    # time axis
    if timestamps is not None and len(timestamps) == n_steps:
        time_axis = timestamps - timestamps[0]
        time_label = "Time (s)"
        print(f"  Time span: {time_axis[0]:.2f} - {time_axis[-1]:.2f} s")
    else:
        time_axis = np.arange(n_steps)
        time_label = "Steps"
        print(f"  Step range: 0 - {n_steps}")
    
    fig = plt.figure(figsize=(16, 10))
    episode_name = os.path.basename(os.path.dirname(hdf5_path))
    fig.suptitle(f'Arm Trajectory Visualization - {episode_name}', 
                 fontsize=14, fontweight='bold')
    
    # subplots layout 3x3 (last for gripper)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # plot 6 joints
    for i in range(n_joints):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        if n_steps > 5000:
            step = max(1, n_steps // 2000)
            plot_indices = np.arange(0, n_steps, step)
            plot_time = time_axis[plot_indices]
            plot_data = joint_pos[plot_indices, i]
            print(f"[INFO] Joint {i+1} downsampled for display: {n_steps} -> {len(plot_indices)}")
        else:
            plot_time = time_axis
            plot_data = joint_pos[:, i]
        
        ax.plot(plot_time, plot_data, 
                color=colors[i % len(colors)], 
                linewidth=1.2,
                alpha=0.9,
                label=f'Joint {i+1}')
        
        ax.set_xlabel(time_label, fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.set_title(f'Joint {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)
        
        min_val = np.min(joint_pos[:, i])
        max_val = np.max(joint_pos[:, i])
        mean_val = np.mean(joint_pos[:, i])
        std_val = np.std(joint_pos[:, i])
        
        info_text = f'Min: {min_val:+.3f}\nMax: {max_val:+.3f}\nMean: {mean_val:+.3f}\nStd: {std_val:.3f}'
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # gripper width
    ax_gripper = fig.add_subplot(gs[2, 2])
    
    if n_steps > 5000:
        step = max(1, n_steps // 2000)
        plot_indices = np.arange(0, n_steps, step)
        plot_time = time_axis[plot_indices]
        plot_gripper = gripper_width[plot_indices]
    else:
        plot_time = time_axis
        plot_gripper = gripper_width
    
    ax_gripper.plot(plot_time, plot_gripper, 
                    color='#E74C3C', 
                    linewidth=1.5,
                    alpha=0.9,
                    label='Gripper')
    
    ax_gripper.set_xlabel(time_label, fontsize=10)
    ax_gripper.set_ylabel('Width (m)', fontsize=10)
    ax_gripper.set_title('Gripper Width', fontsize=12, fontweight='bold')
    ax_gripper.grid(True, alpha=0.3, linestyle='--')
    ax_gripper.legend(loc='best', fontsize=8)
    
    min_gripper = np.min(gripper_width)
    max_gripper = np.max(gripper_width)
    mean_gripper = np.mean(gripper_width)
    std_gripper = np.std(gripper_width)
    
    info_text = f'Min: {min_gripper:.4f}\nMax: {max_gripper:.4f}\nMean: {mean_gripper:.4f}\nStd: {std_gripper:.4f}'
    ax_gripper.text(0.02, 0.98, info_text, 
                    transform=ax_gripper.transAxes,
                    fontsize=7,
                    verticalalignment='top',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_trajectory_detailed(hdf5_path, save_path=None, show=True):
    """
    Plot detailed trajectories (pos/vel/acc)

    Args:
        hdf5_path: path to HDF5 file
        save_path: optional path to save figure
        show: whether to show window
    """
    
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] File not found: {hdf5_path}")
        return
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            joint_pos = f['state/joint/position'][:]
            gripper_width = f['state/joint/gripper_width'][:]
            
            timestamps = None
            if 'observation/rgb_timestamp' in f:
                timestamps = f['observation/rgb_timestamp'][:]
    
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return
    
    n_steps = joint_pos.shape[0]
    n_joints = joint_pos.shape[1]
    
    # time step
    if timestamps is not None and len(timestamps) >= 2:
        dt_array = np.diff(timestamps)
        dt = np.median(dt_array[dt_array < 1.0])
    else:
        dt = 0.01  # default 100 Hz
    
    # velocity & acceleration
    joint_vel = np.diff(joint_pos, axis=0) / dt
    joint_acc = np.diff(joint_vel, axis=0) / dt
    
    gripper_vel = np.diff(gripper_width) / dt
    
    # time axes
    if timestamps is not None:
        time_pos = timestamps
        time_vel = timestamps[:-1]
        time_acc = timestamps[:-2]
    else:
        time_pos = np.arange(n_steps) * dt
        time_vel = np.arange(n_steps - 1) * dt
        time_acc = np.arange(n_steps - 2) * dt
    
    # figure (per joint: pos/vel/acc)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Arm trajectory detailed analysis\nFile: {os.path.basename(hdf5_path)}', 
                 fontsize=14, fontweight='bold')
    
    # subplots
    n_rows = n_joints + 1  # 6 joints + gripper
    n_cols = 3  # pos/vel/acc
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    # joints
    for i in range(n_joints):
        # 位置
        ax1 = plt.subplot(n_rows, n_cols, i * n_cols + 1)
        ax1.plot(time_pos, joint_pos[:, i], color=colors[i], linewidth=1.5)
        ax1.set_ylabel(f'Joint {i+1}\nPos (rad)', fontsize=9)
        ax1.grid(True, alpha=0.3)
        if i == 0:
            ax1.set_title('Position', fontsize=11, fontweight='bold')
        
        # 速度
        ax2 = plt.subplot(n_rows, n_cols, i * n_cols + 2)
        ax2.plot(time_vel, joint_vel[:, i], color=colors[i], linewidth=1.5)
        ax2.set_ylabel(f'Vel (rad/s)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.set_title('Velocity', fontsize=11, fontweight='bold')
        
        # 加速度
        ax3 = plt.subplot(n_rows, n_cols, i * n_cols + 3)
        ax3.plot(time_acc, joint_acc[:, i], color=colors[i], linewidth=1.5)
        ax3.set_ylabel(f'Acc (rad/s²)', fontsize=9)
        ax3.grid(True, alpha=0.3)
        if i == 0:
            ax3.set_title('Acceleration', fontsize=11, fontweight='bold')
    
    # gripper
    i = n_joints
    
    ax1 = plt.subplot(n_rows, n_cols, i * n_cols + 1)
    ax1.plot(time_pos, gripper_width, color='red', linewidth=1.5)
    ax1.set_ylabel('Gripper\nWidth (m)', fontsize=9)
    ax1.set_xlabel('Time (s)', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(n_rows, n_cols, i * n_cols + 2)
    ax2.plot(time_vel, gripper_vel, color='red', linewidth=1.5)
    ax2.set_ylabel('Vel\n(m/s)', fontsize=9)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(n_rows, n_cols, i * n_cols + 3)
    ax3.axis('off')
    
    stats_text = f"Trajectory stats\n\n"
    stats_text += f"Steps: {n_steps}\n"
    stats_text += f"Duration: {time_pos[-1]:.2f} s\n"
    stats_text += f"Sampling: {1/dt:.1f} Hz\n\n"
    
    stats_text += "Max joint velocity:\n"
    for j in range(n_joints):
        max_v = np.max(np.abs(joint_vel[:, j]))
        stats_text += f"  Joint {j+1}: {max_v:.2f} rad/s\n"
    
    stats_text += f"\nGripper max velocity: {np.max(np.abs(gripper_vel)):.3f} m/s"
    
    ax3.text(0.1, 0.9, stats_text, 
             transform=ax3.transAxes,
             fontsize=9,
             verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Detailed figure saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def diagnose_data_quality(joint_pos, gripper_width):
    """Diagnose data quality and detect vibration/spikes"""
    print("\n[INFO] Data quality diagnosis:")
    
    for i in range(joint_pos.shape[1]):
        diff = np.diff(joint_pos[:, i])
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        vibration_ratio = sign_changes / len(diff) * 100
        
        if vibration_ratio > 50:
            print(f"  ⚠️  Joint {i+1}: High-frequency oscillation ({vibration_ratio:.1f}% sign changes)")
        
        max_jump = np.max(np.abs(diff))
        if max_jump > 1.0:
            print(f"  ⚠️  Joint {i+1}: Large jump detected (max: {max_jump:.3f} rad)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Robot Trajectory")
    parser.add_argument("hdf5_path", type=str, help="HDF5 file path")
    parser.add_argument("--save", type=str, default=None, 
                        help="Save figure to path (optional)")
    parser.add_argument("--no_show", action="store_true", 
                        help="Don't show window, only save")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed analysis (velocity & acceleration)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Robot Trajectory Visualization")
    print("=" * 70)
    print(f"File: {args.hdf5_path}")
    print("=" * 70)
    print()
    
    # pre-diagnose
    try:
        with h5py.File(args.hdf5_path, 'r') as f:
            joint_pos = f['state/joint/position'][:]
            gripper_width = f['state/joint/gripper_width'][:]
        
        diagnose_data_quality(joint_pos, gripper_width)
        print()
    except Exception as e:
        print(f"[WARN] Unable to diagnose data: {e}\n")
    
    if args.detailed:
        plot_trajectory_detailed(
            args.hdf5_path, 
            save_path=args.save, 
            show=not args.no_show
        )
    else:
        plot_trajectory(
            args.hdf5_path, 
            save_path=args.save, 
            show=not args.no_show
        )


if __name__ == "__main__":
    main()

