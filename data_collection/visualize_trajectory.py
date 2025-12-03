#!/usr/bin/env python3
"""
可视化轨迹数据工具

功能：读取HDF5文件中的state数据，绘制6个关节 + 1个夹爪的轨迹图
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_trajectory(hdf5_path, save_path=None, show=True):
    """
    绘制轨迹图
    
    Args:
        hdf5_path: HDF5文件路径
        save_path: 保存图片路径（可选）
        show: 是否显示窗口
    """
    
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] 文件不存在: {hdf5_path}")
        return
    
    # 读取数据
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 读取关节位置
            if 'state/joint/position' not in f:
                print(f"[ERROR] 文件中没有 'state/joint/position'")
                return
            
            joint_pos = f['state/joint/position'][:]
            
            # 读取夹爪宽度
            if 'state/joint/gripper_width' not in f:
                print(f"[ERROR] 文件中没有 'state/joint/gripper_width'")
                return
            
            gripper_width = f['state/joint/gripper_width'][:]
            
            # 读取时间戳（如果有）
            timestamps = None
            if 'observation/rgb_timestamp' in f:
                timestamps = f['observation/rgb_timestamp'][:]
    
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return
    
    # 数据信息
    n_steps = joint_pos.shape[0]
    n_joints = joint_pos.shape[1]
    
    print(f"[INFO] 读取数据:")
    print(f"  步数: {n_steps}")
    print(f"  关节数: {n_joints}")
    
    # 检查每个关节的数据范围
    print(f"\n[INFO] 各关节数据范围:")
    for i in range(n_joints):
        min_val = np.min(joint_pos[:, i])
        max_val = np.max(joint_pos[:, i])
        std_val = np.std(joint_pos[:, i])
        print(f"  Joint {i+1}: [{min_val:+.4f}, {max_val:+.4f}], std={std_val:.4f}")
    
    print(f"  Gripper: [{np.min(gripper_width):.4f}, {np.max(gripper_width):.4f}], std={np.std(gripper_width):.4f}")
    
    # 创建时间轴
    if timestamps is not None and len(timestamps) == n_steps:
        # 使用相对时间（从0开始）
        time_axis = timestamps - timestamps[0]
        time_label = "Time (s)"
        print(f"  时间范围: {time_axis[0]:.2f} - {time_axis[-1]:.2f} 秒")
    else:
        time_axis = np.arange(n_steps)
        time_label = "Steps"
        print(f"  步数范围: 0 - {n_steps}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    episode_name = os.path.basename(os.path.dirname(hdf5_path))
    fig.suptitle(f'Arm Trajectory Visualization - {episode_name}', 
                 fontsize=14, fontweight='bold')
    
    # 使用GridSpec创建子图布局（3行3列，最后一个用于夹爪）
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 关节颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # 绘制6个关节
    for i in range(n_joints):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # 检查数据点数量，如果太多则下采样显示
        if n_steps > 5000:
            # 下采样显示（但保持原始数据的特征）
            step = max(1, n_steps // 2000)
            plot_indices = np.arange(0, n_steps, step)
            plot_time = time_axis[plot_indices]
            plot_data = joint_pos[plot_indices, i]
            print(f"[INFO] 关节{i+1}数据点过多，下采样显示: {n_steps} -> {len(plot_indices)}")
        else:
            plot_time = time_axis
            plot_data = joint_pos[:, i]
        
        # 绘制轨迹
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
        
        # 显示统计信息
        min_val = np.min(joint_pos[:, i])
        max_val = np.max(joint_pos[:, i])
        mean_val = np.mean(joint_pos[:, i])
        std_val = np.std(joint_pos[:, i])
        
        # 在图上添加统计信息
        info_text = f'Min: {min_val:+.3f}\nMax: {max_val:+.3f}\nMean: {mean_val:+.3f}\nStd: {std_val:.3f}'
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 绘制夹爪宽度（占据最后一个子图位置）
    ax_gripper = fig.add_subplot(gs[2, 2])
    
    # 下采样（如果需要）
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
    
    # 夹爪统计信息
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
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] 图片已保存到: {save_path}")
    
    # 显示图形
    if show:
        plt.show()
    
    return fig


def plot_trajectory_detailed(hdf5_path, save_path=None, show=True):
    """
    绘制详细轨迹图（包括速度和加速度）
    
    Args:
        hdf5_path: HDF5文件路径
        save_path: 保存图片路径（可选）
        show: 是否显示窗口
    """
    
    if not os.path.exists(hdf5_path):
        print(f"[ERROR] 文件不存在: {hdf5_path}")
        return
    
    # 读取数据
    try:
        with h5py.File(hdf5_path, 'r') as f:
            joint_pos = f['state/joint/position'][:]
            gripper_width = f['state/joint/gripper_width'][:]
            
            timestamps = None
            if 'observation/rgb_timestamp' in f:
                timestamps = f['observation/rgb_timestamp'][:]
    
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return
    
    n_steps = joint_pos.shape[0]
    n_joints = joint_pos.shape[1]
    
    # 计算时间步
    if timestamps is not None and len(timestamps) >= 2:
        dt_array = np.diff(timestamps)
        dt = np.median(dt_array[dt_array < 1.0])
    else:
        dt = 0.01  # 默认100Hz
    
    # 计算速度和加速度
    joint_vel = np.diff(joint_pos, axis=0) / dt
    joint_acc = np.diff(joint_vel, axis=0) / dt
    
    gripper_vel = np.diff(gripper_width) / dt
    
    # 创建时间轴
    if timestamps is not None:
        time_pos = timestamps
        time_vel = timestamps[:-1]
        time_acc = timestamps[:-2]
    else:
        time_pos = np.arange(n_steps) * dt
        time_vel = np.arange(n_steps - 1) * dt
        time_acc = np.arange(n_steps - 2) * dt
    
    # 创建图形（每个关节3个子图：位置、速度、加速度）
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'机械臂轨迹详细分析\n文件: {os.path.basename(hdf5_path)}', 
                 fontsize=14, fontweight='bold')
    
    # 创建子图
    n_rows = n_joints + 1  # 6个关节 + 1个夹爪
    n_cols = 3  # 位置、速度、加速度
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    # 绘制每个关节
    for i in range(n_joints):
        # 位置
        ax1 = plt.subplot(n_rows, n_cols, i * n_cols + 1)
        ax1.plot(time_pos, joint_pos[:, i], color=colors[i], linewidth=1.5)
        ax1.set_ylabel(f'关节{i+1}\n位置(rad)', fontsize=9)
        ax1.grid(True, alpha=0.3)
        if i == 0:
            ax1.set_title('位置', fontsize=11, fontweight='bold')
        
        # 速度
        ax2 = plt.subplot(n_rows, n_cols, i * n_cols + 2)
        ax2.plot(time_vel, joint_vel[:, i], color=colors[i], linewidth=1.5)
        ax2.set_ylabel(f'速度\n(rad/s)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.set_title('速度', fontsize=11, fontweight='bold')
        
        # 加速度
        ax3 = plt.subplot(n_rows, n_cols, i * n_cols + 3)
        ax3.plot(time_acc, joint_acc[:, i], color=colors[i], linewidth=1.5)
        ax3.set_ylabel(f'加速度\n(rad/s²)', fontsize=9)
        ax3.grid(True, alpha=0.3)
        if i == 0:
            ax3.set_title('加速度', fontsize=11, fontweight='bold')
    
    # 绘制夹爪
    i = n_joints
    
    # 夹爪位置
    ax1 = plt.subplot(n_rows, n_cols, i * n_cols + 1)
    ax1.plot(time_pos, gripper_width, color='red', linewidth=1.5)
    ax1.set_ylabel('夹爪\n宽度(m)', fontsize=9)
    ax1.set_xlabel('时间 (秒)', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 夹爪速度
    ax2 = plt.subplot(n_rows, n_cols, i * n_cols + 2)
    ax2.plot(time_vel, gripper_vel, color='red', linewidth=1.5)
    ax2.set_ylabel('速度\n(m/s)', fontsize=9)
    ax2.set_xlabel('时间 (秒)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 第三个子图显示统计信息
    ax3 = plt.subplot(n_rows, n_cols, i * n_cols + 3)
    ax3.axis('off')
    
    # 统计信息
    stats_text = f"轨迹统计信息\n\n"
    stats_text += f"总步数: {n_steps}\n"
    stats_text += f"持续时间: {time_pos[-1]:.2f} 秒\n"
    stats_text += f"采样频率: {1/dt:.1f} Hz\n\n"
    
    stats_text += "关节最大速度:\n"
    for j in range(n_joints):
        max_v = np.max(np.abs(joint_vel[:, j]))
        stats_text += f"  关节{j+1}: {max_v:.2f} rad/s\n"
    
    stats_text += f"\n夹爪最大速度: {np.max(np.abs(gripper_vel)):.3f} m/s"
    
    ax3.text(0.1, 0.9, stats_text, 
             transform=ax3.transAxes,
             fontsize=9,
             verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] 详细图片已保存到: {save_path}")
    
    # 显示图形
    if show:
        plt.show()
    
    return fig


def diagnose_data_quality(joint_pos, gripper_width):
    """诊断数据质量，识别异常"""
    print("\n[INFO] 数据质量诊断:")
    
    for i in range(joint_pos.shape[1]):
        # 检查高频振荡
        diff = np.diff(joint_pos[:, i])
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        vibration_ratio = sign_changes / len(diff) * 100
        
        if vibration_ratio > 50:
            print(f"  ⚠️  Joint {i+1}: 高频振荡 ({vibration_ratio:.1f}% 方向变化)")
        
        # 检查是否有大幅跳变
        max_jump = np.max(np.abs(diff))
        if max_jump > 1.0:
            print(f"  ⚠️  Joint {i+1}: 存在大幅跳变 (最大: {max_jump:.3f} rad)")


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
    
    # 先读取数据进行诊断
    try:
        with h5py.File(args.hdf5_path, 'r') as f:
            joint_pos = f['state/joint/position'][:]
            gripper_width = f['state/joint/gripper_width'][:]
        
        diagnose_data_quality(joint_pos, gripper_width)
        print()
    except Exception as e:
        print(f"[WARN] 无法进行数据诊断: {e}\n")
    
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

