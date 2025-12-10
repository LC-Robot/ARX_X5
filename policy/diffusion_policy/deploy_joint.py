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

# Adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Add path
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

# Import ARX-5 Interface
from arx5_interface import Arx5JointController, JointState, LogLevel, RobotConfigFactory

# Import Camera
from realsense_d435 import RealsenseAPI 

# Import Diffusion Policy utils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.precise_sleep import precise_wait

# Register eval resolver
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception as e:
    pass

class ArxJointEnv:
    def __init__(self, 
                 controller: Arx5JointController, 
                 cameras: RealsenseAPI, 
                 obs_horizon: int = 2):
        self.controller = controller
        self.cameras = cameras
        self.obs_horizon = obs_horizon
        
        # Buffer to store observations for the horizon
        self.obs_buffer = {
            "rgb": [],        
            "robot_joint": [] 
        }
        
        # Get robot config to know limits if needed
        self.robot_config = self.controller.get_robot_config()
        
        # Warmup cameras
        print("[ArxJointEnv] Warming up cameras...")
        for _ in range(10):
            _ = self.cameras.get_rgb()
            time.sleep(0.1)

    def get_obs(self) -> Dict[str, Any]:
        """
        Capture current observation (Joints + Images).
        """
        # 1. Get Camera Image
        rgb = self.cameras.get_rgb() # Shape: (cam_num, H, W, 3)
        
        # 2. Get Robot State
        joint_state = self.controller.get_joint_state()
        joint_pos = np.array(joint_state.pos()) 
        gripper_pos = np.array([joint_state.gripper_pos]) 

        robot_state = np.concatenate([joint_pos, gripper_pos])
        
        # 3. Update Buffer
        self.obs_buffer["rgb"].append(rgb)
        self.obs_buffer["robot_joint"].append(robot_state)
        
        # Maintain horizon size
        if len(self.obs_buffer["rgb"]) > self.obs_horizon:
            self.obs_buffer["rgb"] = self.obs_buffer["rgb"][-self.obs_horizon:]
            self.obs_buffer["robot_joint"] = self.obs_buffer["robot_joint"][-self.obs_horizon:]
            
        # 4. Pad if not enough history
        while len(self.obs_buffer["rgb"]) < self.obs_horizon:
            self.obs_buffer["rgb"].insert(0, rgb)
            self.obs_buffer["robot_joint"].insert(0, robot_state)

        # 5. Construct Output Dictionary
        out_rgb = np.stack(self.obs_buffer["rgb"]) # [T, ...]
        out_agent_pos = np.stack(self.obs_buffer["robot_joint"]) # [T, 7]
        
        obs_dict = {
            'state': out_agent_pos, 
        }
        
        # Handle camera keys
        if out_rgb.ndim == 5: # [T, N_cam, H, W, C]
            for i in range(out_rgb.shape[1]):
                obs_dict[f'camera_{i}'] = out_rgb[:, i, ...]
        else:
            obs_dict['image'] = out_rgb

        return obs_dict

    def exec_action(self, target_joint_pos, target_gripper_width, frequency):
        dt = 1.0 / frequency
        
        cur_joint_state = self.controller.get_joint_state()
        cur_pos = np.array(cur_joint_state.pos())
        
        diff = target_joint_pos - cur_pos
        
        max_step_delta = 0.1 
        diff = np.clip(diff, -max_step_delta, max_step_delta)
        
        safe_target = cur_pos + diff

        cmd = JointState(6) 
        cmd.pos()[:] = safe_target
        
        max_width = self.robot_config.gripper_width
        cmd.gripper_pos = np.clip(target_gripper_width, 0, max_width)
        
        cmd.timestamp = self.controller.get_timestamp() + dt + 0.05
        
        self.controller.set_joint_cmd(cmd)


def get_real_obs_dict_custom(env_obs, shape_meta):
    """
    Convert numpy obs to tensor, handle channel transpose (H,W,C -> C,H,W).
    """
    obs_dict_np = {}
    
    for key, value in env_obs.items():
        if 'camera' in key or 'image' in key:
            # Image: [T, H, W, C] -> [T, C, H, W], Float32, 0-1
            img = value.astype(np.float32) / 255.0
            img = np.transpose(img, (0, 3, 1, 2)) 
            obs_dict_np[key] = img
        elif key == 'state':
            # Proprioception: [T, D]
            obs_dict_np[key] = value.astype(np.float32)
            
    return obs_dict_np


def main():
    ckpt_path = "/home/le/ARX_X5/policy/diffusion_policy/data/epoch=0100-train_loss=0.0338.ckpt"
    max_steps = 1000
    frequency = 10
    robot_can_port = "can0"
    arm_type = 0
    
    steps_per_inference = 8 

    # 1. Load Checkpoint
    print(f"[INFO] Loading checkpoint from {ckpt_path}...")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Instantiate workspace/policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get Policy
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    
    print(f"[INFO] Policy loaded successfully.")

    # 2. Initialize Hardware
    print(f"[INFO] Initializing ARX-5 Joint Controller ({robot_can_port})...")
    
    controller = Arx5JointController("X5", robot_can_port)
    controller.set_log_level(LogLevel.INFO)
    
    print("[INFO] Resetting to home...")
    controller.reset_to_home()
    time.sleep(2.0)
    
    print("[INFO] Initializing Cameras...")
    cameras = RealsenseAPI()
    
    # Wrapper
    n_obs_steps = cfg.n_obs_steps
    env = ArxJointEnv(controller, cameras, obs_horizon=n_obs_steps)
    
    # 3. Main Loop
    print("[INFO] Starting Inference Loop. Press 'q' to exit.")
    
    # Warmup Policy
    with torch.no_grad():
        policy.reset()
        obs = env.get_obs()
        obs_dict_np = get_real_obs_dict_custom(obs, cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        _ = policy.predict_action(obs_dict)
        
    try:
        dt = 1.0 / frequency
        
        while True:
            cycle_start_time = time.monotonic()
            
            # --- 1. Get Observation ---
            obs = env.get_obs()
            
            # Visualization
            if 'camera_0' in obs:
                vis_img = obs['camera_0'][-1]
            elif 'image' in obs:
                vis_img = obs['image'][-1]
            else:
                vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
                
            cv2.imshow("Inference", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --- 2. Inference ---
            with torch.no_grad():
                obs_dict_np = get_real_obs_dict_custom(obs, cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                # Predict Action Chunk
                # Output shape: [Batch, Horizon, Action_Dim] -> [1, T, 7]
                result = policy.predict_action(obs_dict)
                action_chunk = result['action'][0].detach().to('cpu').numpy() 

            # --- 3. Execute Actions (Absolute Position) ---
            print(f"Pred shape: {action_chunk.shape}")
            print(f"Pred action: {action_chunk[0]}")
            
            for i in range(steps_per_inference):
                step_start = time.monotonic()
                
                if i >= len(action_chunk):
                    break
                
                raw_action = action_chunk[i]
                
                target_joints = raw_action[:6]
                target_gripper = raw_action[6]
                
                env.exec_action(target_joints, target_gripper, frequency)
                
                time_elapsed = time.monotonic() - step_start
                sleep_time = dt - time_elapsed
                if sleep_time > 0:
                    precise_wait(sleep_time)
            
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Safe exit: Resetting to home.")
        controller.reset_to_home()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()