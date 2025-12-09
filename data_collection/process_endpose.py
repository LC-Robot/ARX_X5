#!/usr/bin/env python3
"""
Usage:
    python process_endpose.py --task_dir datasets/pick_and_place --output_zarr datasets/pick_and_place/replay_buffer_endpose.zarr
"""

import os
import argparse
import numpy as np
import h5py
import zarr
from glob import glob
from tqdm import tqdm
import gc


def get_episode_metadata(ep_path):
    hdf5_path = os.path.join(ep_path, "data.hdf5")
    
    if not os.path.exists(hdf5_path):
        return None

    meta = {'path': ep_path}
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check required keys for State
        if 'state/end_effector/position' not in f or 'state/end_effector/euler' not in f:
            return None
            
        eef_pos = f['state/end_effector/position']
        T = eef_pos.shape[0]
        state_dim = 6  # 3 position + 3 euler
        
        # Get gripper dimension for State
        if 'state/end_effector/gripper_width' in f:
            state_dim += 1
        
        # Check required keys for Action
        if 'action/end_effector/delta_position' not in f or 'action/end_effector/delta_euler' not in f:
            return None
        
        action_dim = 6  # 3 delta_pos + 3 delta_euler
        if 'action/end_effector/gripper_width' in f:
            action_dim += 1
        
        img_shape = None
        if 'observation/rgb' in f:
            rgb_group = f['observation/rgb']
            if len(rgb_group.keys()) > 0:
                first_frame = rgb_group['0'][:]
                if first_frame.ndim == 4:
                    # [n_cams, H, W, 3] -> Take first camera
                    img_shape = first_frame[0].shape
                else:
                    img_shape = first_frame.shape
        
        # We align by slicing [:-1], so n_steps reduces by 1
        meta['n_steps'] = T - 1  
        meta['action_dim'] = action_dim
        meta['state_dim'] = state_dim
        meta['img_shape'] = img_shape
                
    return meta


def load_and_process_episode(ep_path):
    hdf5_path = os.path.join(ep_path, "data.hdf5")
    
    with h5py.File(hdf5_path, 'r') as f:
        # --- Load State ---
        eef_position = f['state/end_effector/position'][:]  # [T, 3]
        eef_euler = f['state/end_effector/euler'][:]        # [T, 3]
        
        gripper_width = None
        if 'state/end_effector/gripper_width' in f:
            gripper_width = f['state/end_effector/gripper_width'][:]  # [T,]
        
        # --- Load Action ---
        delta_position = f['action/end_effector/delta_position'][:]  # [T, 3]
        delta_euler = f['action/end_effector/delta_euler'][:]        # [T, 3]
        
        action_gripper = None
        if 'action/end_effector/gripper_width' in f:
            action_gripper = f['action/end_effector/gripper_width'][:]  # [T,]
        
        # --- Load RGB ---
        rgb = None
        if 'observation/rgb' in f:
            rgb_group = f['observation/rgb']
            # Sort keys to ensure correct order 0, 1, 2...
            frame_keys = sorted(rgb_group.keys(), key=lambda x: int(x))
            
            frames = []
            for k in frame_keys:
                frame = rgb_group[k][:]
                if frame.ndim == 4:
                    frame = frame[0]  # Take first camera
                frames.append(frame)
            rgb = np.stack(frames, axis=0)

    # --- Construct State Array ---
    # Slice [:-1] to match Action length
    state_pos = eef_position[:-1]
    state_euler = eef_euler[:-1]
    state = np.concatenate([state_pos, state_euler], axis=1)
    
    if gripper_width is not None:
        state_gripper = gripper_width[:-1].reshape(-1, 1)
        state = np.concatenate([state, state_gripper], axis=1)
    
    # --- Construct Action Array ---
    # Slice [:-1] to maintain alignment consistency
    action_delta_pos = delta_position[:-1]
    action_delta_euler = delta_euler[:-1]
    action = np.concatenate([action_delta_pos, action_delta_euler], axis=1)
    
    if action_gripper is not None:
        action_gripper_aligned = action_gripper[:-1].reshape(-1, 1)
        action = np.concatenate([action, action_gripper_aligned], axis=1)
    
    # --- Align RGB ---
    if rgb is not None:
        rgb = rgb[:-1]
    
    return {
        'state': state.astype(np.float32),
        'action': action.astype(np.float32),
        'rgb': rgb
    }


def process_task_directory_streaming(task_dir, output_zarr, use_compression=True):
    """
    [Streaming] Process the entire task directory and convert to Zarr.
    """
    episode_dirs = sorted(glob(os.path.join(task_dir, "episode_*")))
    
    if not episode_dirs:
        print(f"[ERROR] No episode directories found in {task_dir}")
        return None
    
    print(f"[INFO] Found {len(episode_dirs)} episodes")
    
    # ===== Phase 1: Scan Metadata =====
    print("\n[Phase 1] Scanning metadata...")
    valid_episodes = []
    total_steps = 0
    action_dim = None
    state_dim = None
    img_shape = None
    
    for ep_dir in tqdm(episode_dirs, desc="Scanning metadata"):
        meta = get_episode_metadata(ep_dir)
        if meta is not None:
            valid_episodes.append(meta)
            total_steps += meta['n_steps']
            
            if action_dim is None:
                action_dim = meta['action_dim']
                state_dim = meta['state_dim']
                img_shape = meta['img_shape']
    
    if not valid_episodes:
        print("[ERROR] No valid episodes found")
        return None
    
    print(f"\n[INFO] Valid episodes: {len(valid_episodes)}")
    print(f"[INFO] Total steps: {total_steps}")
    print(f"[INFO] Action dim: {action_dim} (delta_pos + delta_euler + gripper)")
    print(f"[INFO] State dim: {state_dim} (abs_pos + abs_euler + gripper)")
    if img_shape:
        print(f"[INFO] Image shape: {img_shape}")
    
    # ===== Phase 2: Create Zarr File =====
    print(f"\n[Phase 2] Creating zarr file: {output_zarr}")
    
    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)
    
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    if use_compression:
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        img_compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    else:
        compressor = None
        img_compressor = None
    
    # Create datasets
    action_arr = data_group.create_dataset(
        'action',
        shape=(total_steps, action_dim),
        chunks=(100, action_dim),
        dtype=np.float32,
        compressor=compressor
    )
    
    state_arr = data_group.create_dataset(
        'state',
        shape=(total_steps, state_dim),
        chunks=(100, state_dim),
        dtype=np.float32,
        compressor=compressor
    )
    
    if img_shape is not None:
        H, W, C = img_shape
        camera_arr = data_group.create_dataset(
            'camera_0',
            shape=(total_steps, H, W, C),
            chunks=(100, H, W, C),
            dtype=np.uint8,
            compressor=img_compressor
        )
    else:
        camera_arr = None
    
    # ===== Phase 3: Stream Write Data =====
    print("\n[Phase 3] Writing data stream...")
    
    current_idx = 0
    episode_ends = []
    
    for ep_meta in tqdm(valid_episodes, desc="Writing data"):
        ep_data = load_and_process_episode(ep_meta['path'])
        
        if ep_data is None:
            continue
        
        ep_len = ep_data['action'].shape[0]
        end_idx = current_idx + ep_len
        
        action_arr[current_idx:end_idx] = ep_data['action']
        state_arr[current_idx:end_idx] = ep_data['state']
        
        if camera_arr is not None and ep_data['rgb'] is not None:
            camera_arr[current_idx:end_idx] = ep_data['rgb']
        
        episode_ends.append(end_idx)
        current_idx = end_idx
        
        del ep_data
        gc.collect()
    
    # Write episode_ends
    meta_group.create_dataset(
        'episode_ends',
        data=np.array(episode_ends, dtype=np.int64),
        compressor=compressor
    )
    
    print(f"\n[SUCCESS] Data saved to {output_zarr}")
    print(f"[INFO] Total episodes: {len(episode_ends)}")
    print(f"[INFO] Total steps: {current_idx}")
    
    return output_zarr


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert collected data to Diffusion Policy Zarr format (End-Effector Control)")
    parser.add_argument("--task_dir", type=str, required=True, 
                        help="Task directory containing episode_* subdirectories")
    parser.add_argument("--output_zarr", type=str, default=None,
                        help="Output zarr path (default: task_dir/replay_buffer_endpose.zarr)")
    parser.add_argument("--no_compression", action="store_true",
                        help="Disable compression")
    return parser.parse_args()


def main():
    args = get_arguments()
    
    if args.output_zarr is None:
        args.output_zarr = os.path.join(args.task_dir, "replay_buffer_endpose.zarr")
    
    print("=" * 70)
    print("Data Conversion: Collection Format -> Diffusion Policy Zarr Format")
    print("[End-Effector Control: State=Absolute Pose, Action=Delta Pose]")
    print("[Using streaming processing to avoid memory overflow]")
    print("=" * 70)
    print(f"Input dir: {args.task_dir}")
    print(f"Output path: {args.output_zarr}")
    print(f"Use Compression: {not args.no_compression}")
    print("=" * 70)
    
    result = process_task_directory_streaming(
        args.task_dir,
        args.output_zarr,
        use_compression=not args.no_compression
    )
    
    if result:
        print("\nConversion complete! You can use this zarr file in Diffusion Policy training config.")


if __name__ == "__main__":
    main()