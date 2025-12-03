#!/usr/bin/env python3
"""
自动删除诊断出的有问题的 episode
"""

import os
import shutil
import argparse
from glob import glob

def clean_bad_episodes(task_dir, bad_episodes, backup=True):
    """
    删除有问题的 episode
    
    Args:
        task_dir: 任务目录
        bad_episodes: 有问题的 episode 列表（例如：['episode_10', 'episode_13']）
        backup: 是否先备份
    """
    if backup:
        backup_dir = task_dir + "_backup"
        if os.path.exists(backup_dir):
            print(f"[WARN] 备份目录已存在: {backup_dir}")
            response = input("是否覆盖备份？(y/n): ")
            if response.lower() != 'y':
                print("取消操作")
                return
            shutil.rmtree(backup_dir)
        
        print(f"[INFO] 备份数据到: {backup_dir}")
        shutil.copytree(task_dir, backup_dir)
        print("[INFO] 备份完成")
    
    print(f"\n[INFO] 开始删除 {len(bad_episodes)} 个有问题的 episode...")
    
    deleted_count = 0
    for ep_name in bad_episodes:
        ep_path = os.path.join(task_dir, ep_name)
        if os.path.exists(ep_path):
            print(f"  删除: {ep_name}")
            shutil.rmtree(ep_path)
            deleted_count += 1
        else:
            print(f"  [SKIP] 不存在: {ep_name}")
    
    print(f"\n[SUCCESS] 已删除 {deleted_count} 个 episode")
    
    # 统计剩余 episode
    remaining_episodes = sorted(glob(os.path.join(task_dir, "episode_*")))
    print(f"[INFO] 剩余 {len(remaining_episodes)} 个 episode")
    
    return len(remaining_episodes)


def main():
    parser = argparse.ArgumentParser(description="清理有问题的 episode")
    parser.add_argument("--task_dir", type=str, required=True, help="任务目录")
    parser.add_argument("--bad_episodes", type=str, nargs='+', required=True,
                        help="要删除的 episode 名称列表")
    parser.add_argument("--no_backup", action="store_true", help="不备份（危险！）")
    parser.add_argument("--force", action="store_true", help="不询问，直接删除")
    args = parser.parse_args()
    
    print("=" * 70)
    print("清理有问题的 Episode")
    print("=" * 70)
    print(f"任务目录: {args.task_dir}")
    print(f"要删除的 episode: {', '.join(args.bad_episodes)}")
    print(f"备份: {'否' if args.no_backup else '是'}")
    print("=" * 70)
    
    if not args.force:
        response = input("\n确认删除这些 episode？(yes/no): ")
        if response.lower() != 'yes':
            print("取消操作")
            return
    
    clean_bad_episodes(
        args.task_dir,
        args.bad_episodes,
        backup=not args.no_backup
    )
    
    print("\n[INFO] 现在可以重新运行 process_joint.py 处理数据")


if __name__ == "__main__":
    main()

