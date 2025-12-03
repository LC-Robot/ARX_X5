#!/usr/bin/env python3
"""
高级数据质量诊断工具

功能：
1. 检查空值、无穷大、异常值
2. 检查轨迹连续性（速度/加速度突变）
3. 检查数据完整性（图像、元数据）
4. 评分系统，筛选优质轨迹
5. 生成详细报告和可视化
"""

import os
import sys
import numpy as np
import h5py
from glob import glob
import argparse
import json
from collections import defaultdict


class EpisodeQualityChecker:
    """单个 Episode 的质量检查器"""
    
    def __init__(self, ep_path, dt=0.01):
        self.ep_path = ep_path
        self.ep_name = os.path.basename(ep_path)
        self.dt = dt  # 控制频率时间步
        
        self.issues = []
        self.warnings = []
        self.stats = {}
        self.score = 100  # 初始满分
        
    def check(self):
        """执行所有检查"""
        hdf5_path = os.path.join(self.ep_path, "data.hdf5")
        
        if not os.path.exists(hdf5_path):
            self.issues.append("HDF5 文件不存在")
            self.score = 0
            return self.get_result()
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # 1. 基础检查
                self._check_data_structure(f)
                
                # 2. 加载数据
                joint_pos, gripper_width, timestamps = self._load_data(f)
                
                if joint_pos is None:
                    return self.get_result()
                
                # 2.5 计算实际时间步（如果有 timestamp）
                actual_dt = self._compute_actual_dt(timestamps, joint_pos.shape[0])
                if actual_dt is not None:
                    self.dt = actual_dt
                    self.stats['actual_dt'] = float(actual_dt)
                    self.stats['actual_frequency'] = float(1.0 / actual_dt)
                
                # 3. 数值检查
                self._check_values(joint_pos, gripper_width)
                
                # 4. 轨迹连续性检查
                self._check_trajectory_continuity(joint_pos, gripper_width)
                
                # 5. 统计信息
                self._compute_statistics(joint_pos, gripper_width)
                
                # 6. 图像数据检查
                self._check_images(f)
                
                # 7. 元数据检查
                self._check_metadata()
                
        except Exception as e:
            self.issues.append(f"读取失败: {e}")
            self.score = 0
        
        return self.get_result()
    
    def _check_data_structure(self, f):
        """检查数据结构完整性"""
        required_fields = [
            'state/joint/position',
            'state/joint/gripper_width',
            'action/joint/position',
            'action/joint/gripper_width',
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in f:
                missing_fields.append(field)
        
        if missing_fields:
            self.issues.append(f"缺少必需字段: {', '.join(missing_fields)}")
            self.score -= 50
    
    def _compute_actual_dt(self, timestamps, n_steps):
        """从 timestamp 计算实际的时间步"""
        if timestamps is None or len(timestamps) < 2:
            return None
        
        # 确保 timestamps 和 joint_pos 长度一致
        if len(timestamps) != n_steps:
            self.warnings.append(f"Timestamp 数量 ({len(timestamps)}) 与步数 ({n_steps}) 不匹配")
            return None
        
        # 计算相邻帧的时间差
        dt_array = np.diff(timestamps)
        
        # 过滤掉异常值（比如暂停采集时的大间隔）
        valid_dt = dt_array[dt_array < 1.0]  # 假设采集频率 > 1Hz
        
        if len(valid_dt) == 0:
            return None
        
        # 使用中位数作为实际 dt（比均值更鲁棒）
        actual_dt = np.median(valid_dt)
        
        return actual_dt
    
    def _load_data(self, f):
        """加载数据"""
        try:
            joint_pos = f['state/joint/position'][:]
            gripper_width = f['state/joint/gripper_width'][:]
            
            # 尝试加载 timestamp（如果有）
            timestamps = None
            if 'observation/rgb_timestamp' in f:
                timestamps = f['observation/rgb_timestamp'][:]
            
            return joint_pos, gripper_width, timestamps
        except Exception as e:
            self.issues.append(f"数据加载失败: {e}")
            return None, None, None
    
    def _check_values(self, joint_pos, gripper_width):
        """检查数值范围和有效性"""
        
        # 检查 NaN
        nan_joints = np.any(np.isnan(joint_pos))
        nan_gripper = np.any(np.isnan(gripper_width))
        
        if nan_joints:
            self.issues.append("关节位置包含 NaN")
            self.score -= 30
        
        if nan_gripper:
            self.issues.append("夹爪宽度包含 NaN")
            self.score -= 20
        
        # 检查 Inf
        inf_joints = np.any(np.isinf(joint_pos))
        inf_gripper = np.any(np.isinf(gripper_width))
        
        if inf_joints:
            self.issues.append("关节位置包含 Inf")
            self.score -= 30
        
        if inf_gripper:
            self.issues.append("夹爪宽度包含 Inf")
            self.score -= 20
        
        # 检查关节角度范围（正常应该在 [-2π, 2π]）
        max_joint = np.max(np.abs(joint_pos))
        if max_joint > 10:
            self.issues.append(f"关节角度异常大: {max_joint:.2e}")
            self.score -= 40
        elif max_joint > 6.5:  # 略大于 2π
            self.warnings.append(f"关节角度偏大: {max_joint:.2f}")
            self.score -= 10
        
        # 检查夹爪范围（应该在 [0, 0.088]）
        min_gripper = np.min(gripper_width)
        max_gripper = np.max(gripper_width)
        
        if min_gripper < -0.01 or max_gripper > 0.1:
            self.issues.append(f"夹爪宽度超出范围: [{min_gripper:.4f}, {max_gripper:.4f}]")
            self.score -= 25
        elif min_gripper < 0 or max_gripper > 0.09:
            self.warnings.append(f"夹爪宽度接近边界: [{min_gripper:.4f}, {max_gripper:.4f}]")
            self.score -= 5
    
    def _check_trajectory_continuity(self, joint_pos, gripper_width):
        """检查轨迹连续性（检测突变）"""
        
        # 计算速度（一阶差分）
        joint_vel = np.diff(joint_pos, axis=0) / self.dt
        gripper_vel = np.diff(gripper_width) / self.dt
        
        # 计算加速度（二阶差分）
        joint_acc = np.diff(joint_vel, axis=0) / self.dt
        gripper_acc = np.diff(gripper_vel) / self.dt
        
        # 检查速度突变（关节速度通常 < 5 rad/s）
        max_joint_vel = np.max(np.abs(joint_vel))
        if max_joint_vel > 10:
            self.issues.append(f"关节速度异常大: {max_joint_vel:.2f} rad/s")
            self.score -= 20
        elif max_joint_vel > 6:
            self.warnings.append(f"关节速度偏大: {max_joint_vel:.2f} rad/s")
            self.score -= 5
        
        # 检查加速度突变（关节加速度通常 < 50 rad/s²）
        max_joint_acc = np.max(np.abs(joint_acc))
        if max_joint_acc > 100:
            self.issues.append(f"关节加速度异常大: {max_joint_acc:.2f} rad/s²")
            self.score -= 15
        elif max_joint_acc > 60:
            self.warnings.append(f"关节加速度偏大: {max_joint_acc:.2f} rad/s²")
            self.score -= 3
        
        # 检查单步突变（相邻帧变化过大）
        max_single_step = np.max(np.abs(np.diff(joint_pos, axis=0)), axis=0)
        sudden_jump_threshold = 0.5  # 单步变化超过 0.5 rad 认为是突变
        
        sudden_jumps = max_single_step > sudden_jump_threshold
        if np.any(sudden_jumps):
            jump_joints = np.where(sudden_jumps)[0]
            self.warnings.append(f"关节 {jump_joints} 存在突变 (Δ > {sudden_jump_threshold})")
            self.score -= 10
        
        # 检查夹爪速度
        max_gripper_vel = np.max(np.abs(gripper_vel))
        if max_gripper_vel > 0.5:  # 夹爪速度通常 < 0.3 m/s
            self.warnings.append(f"夹爪速度偏大: {max_gripper_vel:.2f} m/s")
            self.score -= 3
        
        # 统计
        self.stats['max_joint_velocity'] = float(max_joint_vel)
        self.stats['max_joint_acceleration'] = float(max_joint_acc)
        self.stats['max_gripper_velocity'] = float(max_gripper_vel)
        self.stats['max_single_step_change'] = float(np.max(max_single_step))
    
    def _compute_statistics(self, joint_pos, gripper_width):
        """计算统计信息"""
        
        self.stats['n_steps'] = int(joint_pos.shape[0])
        self.stats['duration'] = float(joint_pos.shape[0] * self.dt)
        self.stats['dt_used'] = float(self.dt)
        
        self.stats['joint_position'] = {
            'shape': joint_pos.shape,
            'mean': float(np.mean(joint_pos)),
            'std': float(np.std(joint_pos)),
            'min': float(np.min(joint_pos)),
            'max': float(np.max(joint_pos)),
        }
        
        self.stats['gripper_width'] = {
            'mean': float(np.mean(gripper_width)),
            'std': float(np.std(gripper_width)),
            'min': float(np.min(gripper_width)),
            'max': float(np.max(gripper_width)),
        }
        
        # 运动范围（关节活动度）
        joint_range = np.max(joint_pos, axis=0) - np.min(joint_pos, axis=0)
        self.stats['joint_range'] = [float(x) for x in joint_range]
        self.stats['avg_joint_range'] = float(np.mean(joint_range))
        
        # 如果运动范围太小，可能是静止数据
        if np.mean(joint_range) < 0.1:
            self.warnings.append(f"关节运动范围很小: {np.mean(joint_range):.3f} rad")
            self.score -= 5
    
    def _check_images(self, f):
        """检查图像数据"""
        if 'observation/rgb' not in f:
            self.warnings.append("缺少图像数据")
            self.score -= 5
            return
        
        try:
            rgb_group = f['observation/rgb']
            n_frames = len(rgb_group.keys())
            
            if n_frames == 0:
                self.issues.append("图像数据为空")
                self.score -= 20
            else:
                # 检查第一帧
                first_frame = rgb_group['0'][:]
                
                # 检查图像是否全黑或全白
                if np.all(first_frame == 0):
                    self.warnings.append("首帧图像全黑")
                    self.score -= 5
                elif np.all(first_frame == 255):
                    self.warnings.append("首帧图像全白")
                    self.score -= 5
                
                self.stats['n_frames'] = n_frames
                self.stats['image_shape'] = first_frame.shape
                
                # 检查帧数和步数是否匹配
                expected_frames = self.stats.get('n_steps', 0)
                if n_frames != expected_frames and expected_frames > 0:
                    self.warnings.append(f"图像帧数 ({n_frames}) 与数据步数 ({expected_frames}) 不匹配")
                    self.score -= 10
                    
        except Exception as e:
            self.warnings.append(f"图像检查失败: {e}")
            self.score -= 5
    
    def _check_metadata(self):
        """检查元数据文件"""
        metadata_path = os.path.join(self.ep_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            self.warnings.append("缺少 metadata.json")
            self.score -= 3
        else:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.stats['metadata'] = metadata
            except Exception as e:
                self.warnings.append(f"metadata 读取失败: {e}")
                self.score -= 3
    
    def get_result(self):
        """返回检查结果"""
        # 确保分数在 0-100 范围内
        self.score = max(0, min(100, self.score))
        
        # 判定质量等级
        if self.score >= 90:
            quality = "优秀"
        elif self.score >= 75:
            quality = "良好"
        elif self.score >= 60:
            quality = "一般"
        elif self.score >= 40:
            quality = "较差"
        else:
            quality = "劣质"
        
        return {
            'ep_name': self.ep_name,
            'ep_path': self.ep_path,
            'score': self.score,
            'quality': quality,
            'issues': self.issues,
            'warnings': self.warnings,
            'stats': self.stats,
            'is_usable': self.score >= 60,  # 60分以上认为可用
        }


def diagnose_dataset(task_dir, dt=0.01, min_score=60, save_report=True):
    """诊断整个数据集"""
    
    episode_dirs = sorted(glob(os.path.join(task_dir, "episode_*")))
    
    if not episode_dirs:
        print(f"[ERROR] 在 {task_dir} 中没有找到 episode 目录")
        return None
    
    print("=" * 80)
    print("🔍 高级数据质量诊断")
    print("=" * 80)
    print(f"数据目录: {task_dir}")
    print(f"Episode 数量: {len(episode_dirs)}")
    print(f"最低可用分数: {min_score}")
    print("=" * 80)
    print()
    
    results = []
    
    # 逐个检查
    for ep_dir in episode_dirs:
        checker = EpisodeQualityChecker(ep_dir, dt=dt)
        result = checker.check()
        results.append(result)
        
        # 实时显示
        ep_name = result['ep_name']
        score = result['score']
        quality = result['quality']
        
        if score >= 90:
            icon = "✅"
        elif score >= 75:
            icon = "🟢"
        elif score >= 60:
            icon = "🟡"
        elif score >= 40:
            icon = "🟠"
        else:
            icon = "🔴"
        
        # 显示实际频率（如果有）
        freq_info = ""
        if 'actual_frequency' in result['stats']:
            actual_freq = result['stats']['actual_frequency']
            freq_info = f" @ {actual_freq:.1f}Hz"
        
        print(f"{icon} {ep_name:15s} | 分数: {score:3d} | 质量: {quality:4s}{freq_info}", end='')
        
        if result['issues']:
            print(f" | ⚠️  {len(result['issues'])} 个严重问题")
        elif result['warnings']:
            print(f" | ⚡ {len(result['warnings'])} 个警告")
        else:
            print()
    
    # 统计汇总
    print("\n" + "=" * 80)
    print("📊 统计汇总")
    print("=" * 80)
    
    scores = [r['score'] for r in results]
    usable = [r for r in results if r['is_usable']]
    excellent = [r for r in results if r['score'] >= 90]
    good = [r for r in results if 75 <= r['score'] < 90]
    fair = [r for r in results if 60 <= r['score'] < 75]
    poor = [r for r in results if 40 <= r['score'] < 60]
    bad = [r for r in results if r['score'] < 40]
    
    print(f"总 Episode 数: {len(results)}")
    print(f"  ✅ 优秀 (≥90):  {len(excellent):3d} 个")
    print(f"  🟢 良好 (75-89): {len(good):3d} 个")
    print(f"  🟡 一般 (60-74): {len(fair):3d} 个")
    print(f"  🟠 较差 (40-59): {len(poor):3d} 个")
    print(f"  🔴 劣质 (<40):   {len(bad):3d} 个")
    print()
    print(f"可用数据 (≥{min_score}分): {len(usable)}/{len(results)} ({100*len(usable)/len(results):.1f}%)")
    print(f"平均分数: {np.mean(scores):.1f}")
    print(f"中位分数: {np.median(scores):.1f}")
    
    # 显示实际采样频率信息
    actual_freqs = [r['stats'].get('actual_frequency') for r in results if 'actual_frequency' in r['stats']]
    if actual_freqs:
        print(f"\n实际采样频率:")
        print(f"  平均: {np.mean(actual_freqs):.1f} Hz")
        print(f"  范围: [{np.min(actual_freqs):.1f}, {np.max(actual_freqs):.1f}] Hz")
        if np.mean(actual_freqs) < 50:
            print(f"  ⚠️  采样频率较低，建议优化采集代码")
    
    # 统计问题类型
    all_issues = []
    all_warnings = []
    for r in results:
        all_issues.extend(r['issues'])
        all_warnings.extend(r['warnings'])
    
    if all_issues:
        print(f"\n❌ 严重问题汇总 ({len(all_issues)} 个):")
        issue_counts = defaultdict(int)
        for issue in all_issues:
            # 提取问题类型（冒号前的部分）
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] += 1
        
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  - {issue_type}: {count} 次")
    
    if all_warnings:
        print(f"\n⚡ 警告汇总 ({len(all_warnings)} 个):")
        warning_counts = defaultdict(int)
        for warning in all_warnings:
            warning_type = warning.split(':')[0] if ':' in warning else warning
            warning_counts[warning_type] += 1
        
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {warning_type}: {count} 次")
    
    # 推荐的 episode
    print("\n" + "=" * 80)
    print("💡 推荐使用的 Episode")
    print("=" * 80)
    
    if usable:
        print(f"\n推荐保留以下 {len(usable)} 个高质量 episode：")
        for r in sorted(usable, key=lambda x: -x['score']):
            print(f"  {r['ep_name']:15s} (分数: {r['score']:3d}, {r['stats'].get('n_steps', 0):4d} 步)")
    else:
        print("⚠️  没有找到质量足够好的 episode！")
    
    # 需要删除的 episode
    unusable = [r for r in results if not r['is_usable']]
    if unusable:
        print(f"\n建议删除以下 {len(unusable)} 个低质量 episode：")
        for r in sorted(unusable, key=lambda x: x['score']):
            issues_str = f" ({', '.join(r['issues'][:2])}...)" if r['issues'] else ""
            print(f"  {r['ep_name']:15s} (分数: {r['score']:3d}){issues_str}")
    
    # 保存报告
    if save_report:
        report_path = os.path.join(task_dir, "quality_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': len(results),
                    'usable': len(usable),
                    'excellent': len(excellent),
                    'good': len(good),
                    'fair': len(fair),
                    'poor': len(poor),
                    'bad': len(bad),
                    'avg_score': float(np.mean(scores)),
                    'median_score': float(np.median(scores)),
                },
                'episodes': results,
            }, f, indent=2)
        print(f"\n📄 详细报告已保存到: {report_path}")
    
    print("=" * 80)
    
    return results


def generate_clean_command(results, min_score=60):
    """生成清理命令"""
    unusable = [r['ep_name'] for r in results if not r['is_usable']]
    
    if not unusable:
        return None
    
    print("\n" + "=" * 80)
    print("🧹 自动清理命令")
    print("=" * 80)
    print("\n复制以下命令来删除低质量 episode：\n")
    
    task_dir = os.path.dirname(results[0]['ep_path'])
    
    cmd = f"python python/data_collection/clean_bad_episodes.py \\\n"
    cmd += f"    --task_dir {task_dir} \\\n"
    cmd += f"    --bad_episodes {' '.join(unusable)}"
    
    print(cmd)
    print("\n" + "=" * 80)
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="高级数据质量诊断")
    parser.add_argument("--task_dir", type=str, required=True, help="任务目录")
    parser.add_argument("--dt", type=float, default=0.01, help="控制时间步（秒）")
    parser.add_argument("--min_score", type=int, default=60, help="最低可用分数")
    parser.add_argument("--no_report", action="store_true", help="不保存报告")
    args = parser.parse_args()
    
    results = diagnose_dataset(
        args.task_dir,
        dt=args.dt,
        min_score=args.min_score,
        save_report=not args.no_report
    )
    
    if results:
        generate_clean_command(results, min_score=args.min_score)


if __name__ == "__main__":
    main()

