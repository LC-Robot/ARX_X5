import sys
import time
import os
import numpy as np
import torch
import dill
import hydra
import cv2
from omegaconf import OmegaConf
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

from arx5_interface import Arx5CartesianController, EEFState, LogLevel
from realsense_d435 import RealsenseAPI
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.precise_sleep import precise_wait

try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception as e:
    pass

class ArxEnv:
    def __init__(self, controller, cameras, obs_horizon=2):
        self.controller = controller
        self.cameras = cameras
        self.obs_horizon = obs_horizon
        self.obs_buffer = {'rgb': [], 'pose': [], 'gripper': []}
        self.robot_config = controller.get_robot_config()
        self.preview_time = 0.3
        
        print("[ArxEnv] Warming up cameras...")
        for _ in range(10):
            _ = self.cameras.get_rgb()
            time.sleep(0.1)

    def get_obs(self) -> Dict[str, Any]:
        rgb = self.cameras.get_rgb()
        eef = self.controller.get_eef_state()
        pose = np.array(eef.pose_6d())
        gripper = np.array([eef.gripper_pos])

        self.obs_buffer['rgb'].append(rgb)
        self.obs_buffer['pose'].append(pose)
        self.obs_buffer['gripper'].append(gripper)
        
        if len(self.obs_buffer['rgb']) > self.obs_horizon:
            for k in self.obs_buffer:
                self.obs_buffer[k] = self.obs_buffer[k][-self.obs_horizon:]
        
        while len(self.obs_buffer['rgb']) < self.obs_horizon:
            for k in self.obs_buffer:
                self.obs_buffer[k].insert(0, self.obs_buffer[k][0])

        out_rgb = np.stack(self.obs_buffer['rgb'])
        out_state = np.concatenate([
            np.stack(self.obs_buffer['pose']),
            np.stack(self.obs_buffer['gripper'])
        ], axis=-1)

        obs_dict = {
            'state': out_state,
        }
        
        if out_rgb.ndim == 5: # [T, N, H, W, C]
            for i in range(out_rgb.shape[1]):
                obs_dict[f'camera_{i}'] = out_rgb[:, i, ...]
        else:
            obs_dict['image'] = out_rgb
            
        return obs_dict

    def exec_action(self, target_pose, target_gripper, duration=0.1):
        cmd = EEFState()
        cmd.pose_6d()[:] = target_pose
        cmd.gripper_pos = np.clip(target_gripper, 0, self.robot_config.gripper_width)
        cmd.timestamp = self.controller.get_timestamp() + duration + 0.05
        
        self.controller.set_eef_cmd(cmd)

    def smooth_move_to_pose(self, target_pose_6d, target_gripper_pos=0.0, duration=3.0, frequency=20):
        current_eef = self.controller.get_eef_state()
        current_pose_6d = np.array(current_eef.pose_6d()).copy()
        current_gripper_pos = current_eef.gripper_pos
        
        num_steps = int(duration * frequency)
        
        start_time = time.monotonic()
        
        for step in range(num_steps + 1):
            t = step / num_steps
            smooth_t = t * t * (3.0 - 2.0 * t)
            
            interpolated_pose = current_pose_6d + smooth_t * (target_pose_6d - current_pose_6d)
            interpolated_gripper = current_gripper_pos + smooth_t * (target_gripper_pos - current_gripper_pos)
            
            current_timestamp = self.controller.get_timestamp()
            eef_cmd = EEFState()
            eef_cmd.pose_6d()[:] = interpolated_pose
            eef_cmd.gripper_pos = interpolated_gripper
            eef_cmd.timestamp = current_timestamp + self.preview_time
            self.controller.set_eef_cmd(eef_cmd)
            
            target_time = start_time + (step + 1) * 1.0 / frequency
            while time.monotonic() < target_time:
                pass

        time.sleep(0.5)

            
#     return obs_dict_np
def get_real_obs_dict_custom(env_obs, shape_meta):
    """Preprocess observation: resize -> normalize -> transpose"""
    obs_dict_np = {}
    
    target_h, target_w = 96, 96
    for key in shape_meta['obs']:
        if 'camera' in key or 'image' in key:
            shape = shape_meta['obs'][key]['shape']
            target_h, target_w = shape[1], shape[2]
            break

    for key, value in env_obs.items():
        if 'camera' in key or 'image' in key:
            T = value.shape[0]
            resized_imgs = []
            for t in range(T):
                img = cv2.resize(value[t], (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_imgs.append(img)
            value = np.stack(resized_imgs)
            
            img = value.astype(np.float32) / 255.0
            img = np.transpose(img, (0, 3, 1, 2))
            obs_dict_np[key] = img
            
        elif key == 'state':
            obs_dict_np[key] = value.astype(np.float32)
            
    return obs_dict_np

def main():
    ckpt_path = "/home/le/ARX_X5/policy/diffusion_policy/data/raw_data_wrist_epoch=0150-train_loss=0.0129.ckpt"
    frequency = 20         
    steps_per_inference = 6
    max_duration = 120        

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model
    device = torch.device('cuda')
    policy.eval().to(device)
    print("[INFO] Model loaded successfully.")

    print("[INFO] Initializing ARX-5...")
    controller = Arx5CartesianController("X5", "can0")
    controller.set_log_level(LogLevel.INFO)
    controller.reset_to_home()
    
    print("[INFO] Initializing cameras...")
    cameras = RealsenseAPI()
    env = ArxEnv(controller, cameras, obs_horizon=cfg.n_obs_steps)

    # move to a better observation pose
    init_pose_6d = np.array([0.2398, 0.0012, 0.2185, 0.0039, 0.8967, 0.0035])
    init_gripper_pos = 0.0
    env.smooth_move_to_pose(init_pose_6d, init_gripper_pos, duration=3.0)
    time.sleep(1.0)

    print("\n" + "="*50)
    print("Ready! Press 'C' in the OpenCV window to start policy.")
    print("Press 'Q' to exit.")
    print("="*50 + "\n")

    dt = 1.0 / frequency
    target_pose = np.array(controller.get_eef_state().pose_6d())

    whole_traj_timestamps = []
    
    try:
        while True:
            obs = env.get_obs()
            
            vis_img = None
            if 'camera_0' in obs: vis_img = obs['camera_0'][-1]
            elif 'image' in obs: vis_img = obs['image'][-1]
            
            if vis_img is not None:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_img, "Ready: Press 'C'", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Diffusion Policy", vis_img)
            
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("[INFO] Policy started!")
                t_start = time.monotonic()
                iter_idx = 0
                
                policy.reset()
                target_pose = np.array(controller.get_eef_state().pose_6d())

                last_step_timestamp = controller.get_timestamp() + env.preview_time - dt 
                
                while True:
                    t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                    obs = env.get_obs()
                    
                    with torch.no_grad():
                        obs_dict = get_real_obs_dict_custom(obs, cfg.task.shape_meta)
                        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        result = policy.predict_action(obs_dict)
                        action = result['action'][0].detach().to('cpu').numpy()

                    print(f"Action shape: {action.shape}")
                    
                    # [修改点 2] 连续时间戳与防滞后逻辑
                    traj = []
                    step_count = min(steps_per_inference, len(action))

                    # 检查当前系统时间，判断我们的时间游标是否因为推理耗时太长而“过期”了
                    current_sys_time = controller.get_timestamp()
                    min_safe_time = current_sys_time + 0.05 # 预留0.05s通信冗余
                    
                    # 如果下一帧理论时间小于当前安全时间，说明发生严重滞后，必须重置时间轴
                    # 否则底层会报 "Timestamp is in the past" 错误
                    if last_step_timestamp + dt < min_safe_time:
                        # print(f"[WARN] Lag detected! Resetting timestamp.") 
                        last_step_timestamp = current_sys_time + env.preview_time - dt

                    for i in range(step_count):
                        raw_action = action[i]
                        delta_pos = raw_action[:3]
                        delta_euler = raw_action[3:6]
                        gripper = raw_action[6]

                        target_pose[:3] += delta_pos
                        target_pose[3:] += delta_euler

                        this_step_ts = last_step_timestamp + (i + 1) * dt

                        eef_cmd = EEFState()
                        eef_cmd.pose_6d()[:] = target_pose
                        eef_cmd.gripper_pos = np.clip(gripper, 0, env.robot_config.gripper_width)
                        eef_cmd.timestamp = this_step_ts

                        whole_traj_timestamps.append(eef_cmd.timestamp)

                        traj.append(eef_cmd)

                    if len(traj) > 0:
                        controller.set_eef_traj(traj)
                        # 更新游标为这一批动作的最后时刻
                        last_step_timestamp = traj[-1].timestamp

                    if vis_img is not None:
                        if cv2.waitKey(1) == ord('q'):
                            raise KeyboardInterrupt

                    # 等待到本批动作应结束的时间点，保持 Python 循环频率与物理时间同步
                    precise_wait(t_cycle_end)
                    
                    iter_idx += steps_per_inference
                    
                    if time.monotonic() - t_start > max_duration:
                        print("[INFO] Timeout, stopping.")
                        break
                
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        print("[INFO] Resetting robot...")
        controller.reset_to_home()
        cv2.destroyAllWindows()

        np.save('whole_traj_timestamps.npy', np.array(whole_traj_timestamps))

if __name__ == "__main__":
    main()