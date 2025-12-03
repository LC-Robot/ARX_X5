#!/usr/bin/env python3
"""
将采集的数据转换为 Diffusion Policy 所需的 zarr 格式

功能：
1. 读取 HDF5/NPY 格式的采集数据
2. 进行 state/action 错位对齐：action[t] = state[t+1]
3. 单相机图像去掉 n_cams 维度
4. 生成符合 diffusion_policy ReplayBuffer 格式的 zarr 文件

【优化】使用流式处理，避免内存溢出

用法：
    python process_joint.py --task_dir datasets/pick_and_place --output_zarr datasets/pick_and_place/replay_buffer.zarr
"""

import os
import argparse
import numpy as np
import h5py
import zarr
from glob import glob
from tqdm import tqdm
import gc


def get_episode_metadata(ep_path, use_hdf5=True):
    """
    获取单个 episode 的元数据（不加载图像数据）
    
    Returns:
        dict: {
            'path': str,
            'n_steps': int,  # 错位对齐后的步数
            'action_dim': int,
            'state_dim': int,
            'img_shape': tuple or None,  # (H, W, 3)
        }
    """
    hdf5_path = os.path.join(ep_path, "data.hdf5")
    npy_path = os.path.join(ep_path, "data.npy")
    
    meta = {'path': ep_path}
    
    try:
        if use_hdf5 and os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, 'r') as f:
                # 获取 state 维度和步数
                if 'state/joint/position' in f:
                    joint_pos = f['state/joint/position']
                    T = joint_pos.shape[0]
                    state_dim = joint_pos.shape[1]
                else:
                    return None
                
                # 获取 gripper 维度
                if 'state/joint/gripper_width' in f:
                    state_dim += 1  # 加上夹爪
                
                # 获取图像信息
                if 'observation/rgb' in f:
                    rgb_group = f['observation/rgb']
                    if len(rgb_group.keys()) > 0:
                        first_frame = rgb_group['0'][:]
                        if first_frame.ndim == 4:
                            # [n_cams, H, W, 3] -> 取第一个相机
                            img_shape = first_frame[0].shape
                        else:
                            img_shape = first_frame.shape
                    else:
                        img_shape = None
                else:
                    img_shape = None
                
                meta['n_steps'] = T - 1  # 错位对齐后少一步
                meta['action_dim'] = state_dim
                meta['state_dim'] = state_dim
                meta['img_shape'] = img_shape
                
        elif os.path.exists(npy_path):
            raw_data = np.load(npy_path, allow_pickle=True).item()
            
            joint_pos = raw_data['state']['joint']['position']
            if isinstance(joint_pos, list):
                joint_pos = np.array(joint_pos)
            T = joint_pos.shape[0]
            state_dim = joint_pos.shape[1]
            
            if 'gripper_width' in raw_data['state']['joint']:
                state_dim += 1
            
            # 获取图像信息
            if 'observation' in raw_data and 'rgb' in raw_data['observation']:
                rgb = raw_data['observation']['rgb']
                if isinstance(rgb, list) and len(rgb) > 0:
                    first_frame = rgb[0]
                    if isinstance(first_frame, np.ndarray):
                        if first_frame.ndim == 4:
                            img_shape = first_frame[0].shape
                        else:
                            img_shape = first_frame.shape
                    else:
                        img_shape = None
                elif isinstance(rgb, np.ndarray) and rgb.shape[0] > 0:
                    if rgb.ndim == 5:
                        img_shape = rgb[0, 0].shape
                    elif rgb.ndim == 4:
                        img_shape = rgb[0].shape
                    else:
                        img_shape = None
                else:
                    img_shape = None
            else:
                img_shape = None
            
            meta['n_steps'] = T - 1
            meta['action_dim'] = state_dim
            meta['state_dim'] = state_dim
            meta['img_shape'] = img_shape
            
            # 释放内存
            del raw_data
            gc.collect()
        else:
            return None
            
    except Exception as e:
        print(f"[WARN] 无法读取 {ep_path}: {e}")
        return None
    
    return meta


def load_and_process_episode(ep_path, use_hdf5=True):
    """
    加载并处理单个 episode（进行 state/action 错位对齐）
    
    Returns:
        dict: {
            'state': np.ndarray [T-1, state_dim],
            'action': np.ndarray [T-1, action_dim],
            'rgb': np.ndarray [T-1, H, W, 3] or None,
        }
    """
    hdf5_path = os.path.join(ep_path, "data.hdf5")
    npy_path = os.path.join(ep_path, "data.npy")
    
    joint_positions = None
    gripper_width = None
    rgb = None
    
    try:
        if use_hdf5 and os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, 'r') as f:
                joint_positions = f['state/joint/position'][:]
                
                if 'state/joint/gripper_width' in f:
                    gripper_width = f['state/joint/gripper_width'][:]
                
                if 'observation/rgb' in f:
                    rgb_group = f['observation/rgb']
                    n_frames = len(rgb_group.keys())
                    frame_keys = sorted(rgb_group.keys(), key=lambda x: int(x))
                    
                    frames = []
                    for k in frame_keys:
                        frame = rgb_group[k][:]
                        if frame.ndim == 4:
                            frame = frame[0]  # 取第一个相机
                        frames.append(frame)
                    rgb = np.stack(frames, axis=0)
                    
        elif os.path.exists(npy_path):
            raw_data = np.load(npy_path, allow_pickle=True).item()
            
            joint_positions = raw_data['state']['joint']['position']
            if isinstance(joint_positions, list):
                joint_positions = np.array(joint_positions)
            
            if 'gripper_width' in raw_data['state']['joint']:
                gripper_width = raw_data['state']['joint']['gripper_width']
                if isinstance(gripper_width, list):
                    gripper_width = np.array(gripper_width)
            
            if 'observation' in raw_data and 'rgb' in raw_data['observation']:
                rgb = raw_data['observation']['rgb']
                if isinstance(rgb, list):
                    rgb = np.stack(rgb, axis=0)
                if rgb.ndim == 5:
                    rgb = rgb[:, 0]  # 取第一个相机
            
            del raw_data
            gc.collect()
        else:
            return None
            
    except Exception as e:
        print(f"[ERROR] 加载 {ep_path} 失败: {e}")
        return None
    
    # State/Action 错位对齐
    state = joint_positions[:-1]
    action = joint_positions[1:]
    
    if gripper_width is not None:
        state_gripper = gripper_width[:-1].reshape(-1, 1)
        action_gripper = gripper_width[1:].reshape(-1, 1)
        state = np.concatenate([state, state_gripper], axis=1)
        action = np.concatenate([action, action_gripper], axis=1)
    
    # RGB 也要去掉最后一帧
    if rgb is not None:
        rgb = rgb[:-1]
    
    return {
        'state': state.astype(np.float32),
        'action': action.astype(np.float32),
        'rgb': rgb
    }


def process_task_directory_streaming(task_dir, output_zarr, use_hdf5=True, use_compression=True):
    """
    【流式处理】处理整个任务目录，转换为 zarr 格式
    
    不会一次性加载所有数据到内存
    """
    # 查找所有 episode 目录
    episode_dirs = sorted(glob(os.path.join(task_dir, "episode_*")))
    
    if not episode_dirs:
        print(f"[ERROR] 在 {task_dir} 中没有找到 episode 目录")
        return None
    
    print(f"[INFO] 找到 {len(episode_dirs)} 个 episode")
    
    # ===== 第一遍：扫描元数据 =====
    print("\n[Phase 1] 扫描元数据...")
    valid_episodes = []
    total_steps = 0
    action_dim = None
    state_dim = None
    img_shape = None
    
    for ep_dir in tqdm(episode_dirs, desc="扫描元数据"):
        meta = get_episode_metadata(ep_dir, use_hdf5=use_hdf5)
        if meta is not None:
            valid_episodes.append(meta)
            total_steps += meta['n_steps']
            
            if action_dim is None:
                action_dim = meta['action_dim']
                state_dim = meta['state_dim']
                img_shape = meta['img_shape']
    
    if not valid_episodes:
        print("[ERROR] 没有有效的 episode")
        return None
    
    print(f"\n[INFO] 有效 episode: {len(valid_episodes)}")
    print(f"[INFO] 总步数: {total_steps}")
    print(f"[INFO] Action 维度: {action_dim}")
    print(f"[INFO] State 维度: {state_dim}")
    if img_shape:
        print(f"[INFO] 图像尺寸: {img_shape}")
    
    # ===== 创建 zarr 文件 =====
    print(f"\n[Phase 2] 创建 zarr 文件: {output_zarr}")
    
    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)
    
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # 压缩器
    if use_compression:
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        img_compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    else:
        compressor = None
        img_compressor = None
    
    # 创建数据集
    action_arr = data_group.create_dataset(
        'action',
        shape=(total_steps, action_dim),
        chunks=(1, action_dim),
        dtype=np.float32,
        compressor=compressor
    )
    
    state_arr = data_group.create_dataset(
        'state',
        shape=(total_steps, state_dim),
        chunks=(1, state_dim),
        dtype=np.float32,
        compressor=compressor
    )
    
    if img_shape is not None:
        H, W, C = img_shape
        camera_arr = data_group.create_dataset(
            'camera_0',
            shape=(total_steps, H, W, C),
            chunks=(1, H, W, C),
            dtype=np.uint8,
            compressor=img_compressor
        )
    else:
        camera_arr = None
    
    # ===== 第二遍：流式写入数据 =====
    print("\n[Phase 3] 流式写入数据...")
    
    current_idx = 0
    episode_ends = []
    
    for ep_meta in tqdm(valid_episodes, desc="写入数据"):
        # 加载单个 episode
        ep_data = load_and_process_episode(ep_meta['path'], use_hdf5=use_hdf5)
        
        if ep_data is None:
            continue
        
        ep_len = ep_data['action'].shape[0]
        end_idx = current_idx + ep_len
        
        # 写入 action
        action_arr[current_idx:end_idx] = ep_data['action']
        
        # 写入 state
        state_arr[current_idx:end_idx] = ep_data['state']
        
        # 写入图像
        if camera_arr is not None and ep_data['rgb'] is not None:
            camera_arr[current_idx:end_idx] = ep_data['rgb']
        
        episode_ends.append(end_idx)
        current_idx = end_idx
        
        # 释放内存
        del ep_data
        gc.collect()
    
    # 写入 episode_ends
    meta_group.create_dataset(
        'episode_ends',
        data=np.array(episode_ends, dtype=np.int64),
        compressor=compressor
    )
    
    print(f"\n[SUCCESS] 数据已保存到 {output_zarr}")
    print(f"[INFO] Episode 数量: {len(episode_ends)}")
    print(f"[INFO] 总步数: {current_idx}")
    
    return output_zarr


def inspect_zarr(zarr_path):
    """查看 zarr 文件的内容"""
    print(f"\n[INFO] 检查 zarr 文件: {zarr_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    print("\n结构:")
    def print_tree(group, prefix=""):
        for key in group.keys():
            item = group[key]
            if isinstance(item, zarr.Array):
                print(f"{prefix}{key}: {item.shape} {item.dtype}")
            else:
                print(f"{prefix}{key}/")
                print_tree(item, prefix + "  ")
    
    print_tree(root)
    
    # 验证 state/action 对齐
    if 'data/state' in root and 'data/action' in root:
        state = root['data/state']
        action = root['data/action']
        episode_ends = root['meta/episode_ends'][:]
        
        print(f"\n[验证] State shape: {state.shape}")
        print(f"[验证] Action shape: {action.shape}")
        print(f"[验证] Episode ends: {episode_ends[:5]}..." if len(episode_ends) > 5 else f"[验证] Episode ends: {episode_ends}")
        
        # 检查第一个 episode
        if len(episode_ends) > 0:
            end_idx = episode_ends[0]
            
            # 只读取前几个样本验证
            state_sample = state[:min(3, end_idx)]
            action_sample = action[:min(3, end_idx)]
            
            print(f"\n[验证] Episode 0 的前几步:")
            print(f"  State[0]:  {state_sample[0][:4]}...")
            print(f"  Action[0]: {action_sample[0][:4]}...")
            if len(state_sample) > 1:
                print(f"  State[1]:  {state_sample[1][:4]}...")
                
                # 验证 action[t] ≈ state[t+1]
                diff = np.abs(action_sample[0] - state_sample[1]).mean()
                print(f"\n[验证] |action[0] - state[1]| 平均误差: {diff:.6f}")
                if diff < 1e-5:
                    print("[验证] ✅ State/Action 对齐正确！")
                else:
                    print("[验证] ⚠️ State/Action 可能未正确对齐")


def get_arguments():
    parser = argparse.ArgumentParser(description="将采集数据转换为 Diffusion Policy zarr 格式（流式处理）")
    parser.add_argument("--task_dir", type=str, required=True, 
                        help="任务目录路径，包含 episode_* 子目录")
    parser.add_argument("--output_zarr", type=str, default=None,
                        help="输出 zarr 路径（默认为 task_dir/replay_buffer.zarr）")
    parser.add_argument("--use_npy", action="store_true",
                        help="优先使用 NPY 文件而非 HDF5")
    parser.add_argument("--no_compression", action="store_true",
                        help="不使用压缩")
    parser.add_argument("--inspect", action="store_true",
                        help="只检查已有的 zarr 文件，不进行转换")
    return parser.parse_args()


def main():
    args = get_arguments()
    
    # 默认输出路径
    if args.output_zarr is None:
        args.output_zarr = os.path.join(args.task_dir, "replay_buffer.zarr")
    
    if args.inspect:
        # 只检查现有 zarr
        if os.path.exists(args.output_zarr):
            inspect_zarr(args.output_zarr)
        else:
            print(f"[ERROR] zarr 文件不存在: {args.output_zarr}")
        return
    
    print("=" * 70)
    print("数据转换：采集格式 → Diffusion Policy zarr 格式")
    print("【使用流式处理，避免内存溢出】")
    print("=" * 70)
    print(f"输入目录: {args.task_dir}")
    print(f"输出路径: {args.output_zarr}")
    print(f"使用 HDF5: {not args.use_npy}")
    print(f"使用压缩: {not args.no_compression}")
    print("=" * 70)
    
    # 执行转换（流式处理）
    result = process_task_directory_streaming(
        args.task_dir,
        args.output_zarr,
        use_hdf5=not args.use_npy,
        use_compression=not args.no_compression
    )
    
    if result:
        print("\n" + "=" * 70)
        inspect_zarr(args.output_zarr)
        print("=" * 70)
        print("\n转换完成！你可以在 Diffusion Policy 训练配置中使用这个 zarr 文件。")


if __name__ == "__main__":
    main()
