from typing import Dict, List
import torch
import numpy as np
import zarr
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class ArxImageDataset(BaseImageDataset): 
    def __init__(self,
            shape_meta: dict,
            zarr_path: str,
            horizon: int = 16,
            pad_before: int = 0,
            pad_after: int = 0,
            n_obs_steps: int = None,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
        ):
        
        # 加载 zarr 数据
        print(f"[ArxImageDataset] 加载数据: {zarr_path}")
        assert os.path.exists(zarr_path), f"zarr 路径不存在: {zarr_path}"
        
        # 直接从 zarr 创建 ReplayBuffer
        replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        
        print(f"[ArxImageDataset] 数据集信息:")
        print(f"  - Episode 数量: {replay_buffer.n_episodes}")
        print(f"  - 总步数: {replay_buffer.n_steps}")
        
        # 解析 shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        print(f"  - RGB keys: {rgb_keys}")
        print(f"  - Low-dim keys: {lowdim_keys}")
        
        # 设置 key_first_k（只取前 n_obs_steps 个观测）
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        
        # 创建训练/验证划分
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        
        # 限制训练 episode 数量
        if max_train_episodes is not None:
            train_indices = np.where(train_mask)[0]
            if len(train_indices) > max_train_episodes:
                train_indices = train_indices[:max_train_episodes]
                train_mask = np.zeros_like(train_mask)
                train_mask[train_indices] = True
        
        # 创建 sampler
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )
        
        # 保存属性
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        print(f"  - 训练样本数: {len(sampler)}")
    
    def get_validation_dataset(self):
        """返回验证集"""
        val_set = ArxImageDataset.__new__(ArxImageDataset)
        val_set.replay_buffer = self.replay_buffer
        val_set.shape_meta = self.shape_meta
        val_set.rgb_keys = self.rgb_keys
        val_set.lowdim_keys = self.lowdim_keys
        val_set.n_obs_steps = self.n_obs_steps
        val_set.train_mask = self.train_mask
        val_set.val_mask = self.val_mask
        val_set.horizon = self.horizon
        val_set.pad_before = self.pad_before
        val_set.pad_after = self.pad_after
        
        key_first_k = dict()
        if self.n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = self.n_obs_steps
        
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k=key_first_k
        )
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """获取数据归一化器"""
        normalizer = LinearNormalizer()
        
        # Action 归一化
        action_data = self.replay_buffer['action'][:]
        action_stats = array_to_stats(action_data)
        normalizer['action'] = get_range_normalizer_from_stat(action_stats)
        
        # State 归一化（如果存在）
        if 'state' in self.replay_buffer.keys():
            state_data = self.replay_buffer['state'][:]
            state_stats = array_to_stats(state_data)
            normalizer['state'] = get_range_normalizer_from_stat(state_stats)
        
        # RGB 图像归一化（固定范围 [0, 1]）
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        return normalizer
    
    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        # 采样一个序列
        data = self.sampler.sample_sequence(idx)
        
        # 只取前 n_obs_steps 个观测
        T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)
        
        obs_dict = dict()
        
        # 处理 RGB 图像
        for key in self.rgb_keys:
            if key in data:
                # 原始格式: [T, H, W, C]
                # 目标格式: [T, C, H, W], float32, [0, 1]
                img = data[key][T_slice]
                img = np.moveaxis(img, -1, 1)  # [T, H, W, C] -> [T, C, H, W]
                img = img.astype(np.float32) / 255.0
                obs_dict[key] = img
                del data[key]
        
        # 处理 low_dim 数据（state）
        for key in self.lowdim_keys:
            if key in data:
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]
        
        # 如果 lowdim 里有 'state' 但 shape_meta 里用的是其他名字
        # 需要做映射
        if 'state' in data and 'state' not in obs_dict:
            for key in self.lowdim_keys:
                if key not in obs_dict:
                    obs_dict[key] = data['state'][T_slice].astype(np.float32)
                    break
        
        # 构建返回数据
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        
        return torch_data


def test():
    """测试数据集加载"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, required=True)
    args = parser.parse_args()
    
    shape_meta = {
        'obs': {
            'camera_0': {
                'shape': [3, 480, 640],
                'type': 'rgb'
            },
            'state': {
                'shape': [7],
                'type': 'low_dim'
            }
        },
        'action': {
            'shape': [7]
        }
    }
    
    dataset = ArxImageDataset(
        shape_meta=shape_meta,
        zarr_path=args.zarr_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2
    )
    
    print(f"\n数据集长度: {len(dataset)}")
    
    # 测试获取一个样本
    sample = dataset[0]
    print(f"\n样本结构:")
    print(f"  obs keys: {list(sample['obs'].keys())}")
    for k, v in sample['obs'].items():
        print(f"    {k}: {v.shape}, dtype={v.dtype}")
    print(f"  action: {sample['action'].shape}, dtype={sample['action'].dtype}")


if __name__ == "__main__":
    test()

