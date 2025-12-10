#!/usr/bin/env python3
"""
Usage1: python data_clean.py /path/to/dataset/task_name
Usage2: python data_clean.py /path/to/dataset/task_name/episode_0/data.hdf5
"""

import os
import shutil
import argparse
import numpy as np
import h5py
from scipy.signal import savgol_filter

WINDOW_LENGTH_POS = 15
WINDOW_LENGTH_ROT = 21
POLY_ORDER = 3

def smooth_trajectory(input_path):
    print(f"[INFO] Processing: {input_path}")

    try:
        with h5py.File(input_path, "r+") as f:
            pos_key = "state/end_effector/position"
            rot_key = "state/end_effector/euler"

            if pos_key not in f or rot_key not in f:
                print(f"[ERROR] Keys not found in {input_path}")
                return

            pos_data = f[pos_key][:]   # [N, 3]
            rot_data = f[rot_key][:]   # [N, 3]
            
            n_steps = pos_data.shape[0]
            print(f"[INFO] Data length: {n_steps} steps")

            # adapt window sizes to sequence length and ensure odd
            wl_pos = min(WINDOW_LENGTH_POS, n_steps if n_steps % 2 == 1 else n_steps - 1)
            wl_rot = min(WINDOW_LENGTH_ROT, n_steps if n_steps % 2 == 1 else n_steps - 1)
            
            if wl_pos <= POLY_ORDER or wl_rot <= POLY_ORDER:
                print("[WARN] Data too short for filtering. Skipping.")
                return

            # smooth position (XYZ)
            print("[INFO] Smoothing position (XYZ)...")
            pos_smooth = np.zeros_like(pos_data)
            for i in range(3):
                pos_smooth[:, i] = savgol_filter(pos_data[:, i], window_length=wl_pos, polyorder=POLY_ORDER)

            # smooth orientation (RPY) with unwrap to avoid 2pi jumps
            print("[INFO] Smoothing orientation (RPY)...")
            rot_smooth = np.zeros_like(rot_data)
            for i in range(3):
                unwrapped = np.unwrap(rot_data[:, i])
                rot_smooth[:, i] = savgol_filter(unwrapped, window_length=wl_rot, polyorder=POLY_ORDER)

            # overwrite datasets in-place; keep all other datasets unchanged
            del f[pos_key]
            del f[rot_key]
            f.create_dataset(pos_key, data=pos_smooth)
            f.create_dataset(rot_key, data=rot_smooth)

            print("[SUCCESS] Data smoothed and saved in place.")

    except Exception as e:
        print(f"[ERROR] {e}")


def process_folder(root_dir):
    """
    Traverse a directory, find episode_* folders, and smooth their data.hdf5 in place.
    """
    if not os.path.isdir(root_dir):
        print(f"[ERROR] {root_dir} is not a directory")
        return

    episodes = [d for d in os.listdir(root_dir) if d.startswith("episode_")]
    if not episodes:
        print(f"[WARN] No episode_* folders found in {root_dir}")
        return

    episodes = sorted(episodes)
    print(f"[INFO] Found {len(episodes)} episodes in {root_dir}")

    for ep in episodes:
        h5_path = os.path.join(root_dir, ep, "data.hdf5")
        if os.path.isfile(h5_path):
            smooth_trajectory(h5_path)
        else:
            print(f"[WARN] data.hdf5 not found in {ep}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Input HDF5 file or folder containing episode_* subfolders")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        process_folder(args.path)
    else:
        smooth_trajectory(args.path)