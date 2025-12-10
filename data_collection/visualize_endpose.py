#!/usr/bin/env python3
"""
Advanced End-Effector Trajectory Analyzer
-----------------------------------------
Visualizes Position (XYZ) and Orientation (RPY) from HDF5 data.
Performs data quality assessment based on:
1. Continuity (handling Euler angle wrapping)
2. Smoothness (Minimizing Jerk)
3. High-frequency oscillation detection (with noise deadzones)
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Configuration ---
# Threshold below which changes are considered sensor noise (meters for Pos, radians for Rot)
NOISE_FLOOR_POS = 1e-4  # 0.1 mm
NOISE_FLOOR_ROT = 1e-3  # ~0.05 degrees

# Threshold for "Bad" Jerk (heuristic, depends on robot speed)
# If Mean Absolute Jerk exceeds this, the motion is considered non-smooth.
JERK_THRESHOLD_POS = 100.0 
JERK_THRESHOLD_ROT = 50.0

def analyze_smoothness(data, dt, label, is_angle=False):
    """
    Calculates derivatives and evaluates smoothness/jitter.
    
    Args:
        data: 1D numpy array of trajectory data.
        dt: Time interval between steps (seconds).
        label: Name of the dimension (e.g., "X", "Yaw").
        is_angle: Boolean, if True, applies unwrapping.
    
    Returns:
        dict: Statistics containing mean_jerk, oscillation_ratio, etc.
    """
    # 1. Preprocessing: Unwrap angles to remove 2*pi jumps
    if is_angle:
        data = np.unwrap(data)

    # 2. Derivative Calculation (Finite Differences)
    # Velocity (1st derivative), Acceleration (2nd), Jerk (3rd)
    vel = np.gradient(data, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)

    # 3. Jitter Metric A: Mean Absolute Jerk (Physical Smoothness)
    # Lower is better. High jerk = jerky/shaky motion.
    mean_jerk = np.mean(np.abs(jerk))

    # 4. Jitter Metric B: Direction Flips (Oscillation)
    # We only count a flip if the movement magnitude is larger than the noise floor.
    # This prevents counting sensor static noise as "high frequency jitter".
    diff = np.diff(data)
    noise_floor = NOISE_FLOOR_ROT if is_angle else NOISE_FLOOR_POS
    
    # Filter only significant movements
    significant_moves = diff[np.abs(diff) > noise_floor]
    
    if len(significant_moves) > 1:
        # Count how many times the sign (+/-) changes
        sign_changes = np.sum(np.diff(np.sign(significant_moves)) != 0)
        oscillation_ratio = (sign_changes / len(significant_moves)) * 100
    else:
        oscillation_ratio = 0.0

    return {
        "mean_jerk": mean_jerk,
        "oscillation_ratio": oscillation_ratio,
        "data_unwrapped": data, # Return processed data for plotting
        "max_val": np.max(data),
        "min_val": np.min(data)
    }

def print_report(labels, stats_list, is_angle):
    """Prints a formatted table of the analysis results."""
    print("-" * 80)
    kind = "Orientation (RPY)" if is_angle else "Position (XYZ)"
    print(f"Analysis Report: {kind}")
    print("-" * 80)
    print(f"{'Dim':<6} | {'Jerk Score':<12} | {'Oscillation %':<15} | {'Status'}")
    print("-" * 80)

    jerk_thresh = JERK_THRESHOLD_ROT if is_angle else JERK_THRESHOLD_POS

    for i, label in enumerate(labels):
        s = stats_list[i]
        jerk = s['mean_jerk']
        osc = s['oscillation_ratio']
        
        # Determine Status
        issues = []
        if jerk > jerk_thresh:
            issues.append("Rough Motion")
        if osc > 40.0: # If direction flips more than 40% of the time during movement
            issues.append("High Freq Jitter")
        
        status_str = ", ".join(issues) if issues else "OK (Smooth)"
        
        print(f"{label:<6} | {jerk:<12.2f} | {osc:<14.1f}% | {status_str}")
    print("")

def plot_trajectory(hdf5_path, save_path=None, no_show=False):
    """Loads data, analyzes it, and visualizes it."""
    
    # --- Load Data ---
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] File not found: {hdf5_path}")
        return

    try:
        with h5py.File(hdf5_path, "r") as f:
            # Modify these keys based on your specific HDF5 structure
            # Common structures: 'state/end_effector_pose', 'action', etc.
            if "state/end_effector/position" not in f:
                print("[ERROR] Key 'state/end_effector/position' not found.")
                return
            
            pos = f["state/end_effector/position"][:]  # Shape: [N, 3]
            euler = f["state/end_effector/euler"][:]    # Shape: [N, 3]
            
            # Try to get timestamps, otherwise assume 50Hz (0.02s)
            if "observation/rgb_timestamp" in f:
                ts = f["observation/rgb_timestamp"][:]
                # Handle cases where timestamp might be relative or absolute
                ts = ts - ts[0]
                # Calculate average dt
                dt = np.mean(np.diff(ts)) if len(ts) > 1 else 0.02
                # Sanity check for dt
                if dt <= 0 or np.isnan(dt): dt = 0.02
            else:
                ts = np.arange(len(pos)) * 0.02
                dt = 0.02
                
    except Exception as e:
        print(f"[ERROR] Failed to load HDF5: {e}")
        return

    print(f"[INFO] Loaded {len(pos)} steps. Estimated sampling rate: {1/dt:.1f} Hz")

    # --- Analysis Phase ---
    labels_pos = ["X", "Y", "Z"]
    labels_rot = ["Roll", "Pitch", "Yaw"]
    
    pos_stats = []
    rot_stats = []

    # Analyze Position
    for i in range(3):
        res = analyze_smoothness(pos[:, i], dt, labels_pos[i], is_angle=False)
        pos_stats.append(res)

    # Analyze Rotation
    for i in range(3):
        res = analyze_smoothness(euler[:, i], dt, labels_rot[i], is_angle=True)
        rot_stats.append(res)

    # Print Text Report
    print_report(labels_pos, pos_stats, is_angle=False)
    print_report(labels_rot, rot_stats, is_angle=True)

    # --- Visualization Phase ---
    fig = plt.figure(figsize=(16, 9))
    filename = os.path.basename(hdf5_path)
    fig.suptitle(f'Trajectory Quality Analysis: {filename}\n(Dt={dt:.3f}s)', fontsize=14, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    # Helper to plot one subplot
    def add_subplot(row, col, time_axis, data, stats, name, unit, color):
        ax = fig.add_subplot(gs[row, col])
        ax.plot(time_axis, data, color=color, linewidth=1.5, alpha=0.9)
        
        # Add a small smoothing comparison (optional, shows trend)
        # ax.plot(time_axis, data, color='k', alpha=0.2, linewidth=3) 

        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Annotation Box
        info_text = (f"Range: [{stats['min_val']:.2f}, {stats['max_val']:.2f}]\n"
                     f"Jerk: {stats['mean_jerk']:.2f}\n"
                     f"Oscillation: {stats['oscillation_ratio']:.1f}%")
        
        # Color code the box based on quality
        box_color = 'wheat'
        if stats['mean_jerk'] > (JERK_THRESHOLD_ROT if unit=='rad' else JERK_THRESHOLD_POS):
            box_color = '#ffcccc' # Redish for bad data

        ax.text(0.03, 0.95, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    # Plot Position (Row 0)
    colors_pos = ['#e74c3c', '#27ae60', '#2980b9'] # R, G, B
    for i in range(3):
        # Use the unwrapped/processed data from stats
        data_to_plot = pos_stats[i]['data_unwrapped']
        add_subplot(0, i, ts, data_to_plot, pos_stats[i], labels_pos[i], "m", colors_pos[i])

    # Plot Orientation (Row 1)
    colors_rot = ['#e67e22', '#8e44ad', '#16a085'] # Orange, Purple, Teal
    for i in range(3):
        # Use the unwrapped/processed data from stats
        data_to_plot = rot_stats[i]['data_unwrapped']
        add_subplot(1, i, ts, data_to_plot, rot_stats[i], labels_rot[i], "rad", colors_rot[i])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")

    if not no_show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Robot Trajectory Jitter Analysis")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 data file")
    parser.add_argument("--save", type=str, default=None, help="Save the plot to this path (e.g., analysis.png)")
    parser.add_argument("--no_show", action="store_true", help="Do not open the GUI window")
    args = parser.parse_args()

    print("=" * 80)
    print(" JITTER & DATA QUALITY ANALYZER ")
    print("=" * 80)
    
    plot_trajectory(args.hdf5_path, args.save, args.no_show)

if __name__ == "__main__":
    main()