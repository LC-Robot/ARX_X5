import cv2
import os
import h5py
import time
import numpy as np
import sys
import os
from PIL import Image
from type import get_numpy, to_numpy
from realsense_d435 import RealsenseAPI
# Adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Add path
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel

class DataCollector:
    def __init__(self, controller: Arx5CartesianController, cameras: RealsenseAPI, is_image_encode: bool = False, *args, **kwargs):
        self.controller: Arx5CartesianController = controller
        self.cameras: RealsenseAPI = cameras
        self.is_image_encode = is_image_encode
        self.data_dict = self.get_empty_data_dict()
        self.device = 'cpu'

    def get_empty_data_dict(self):
        data_dict = {
            "action": {
                "end_effector": {
                    "delta_orientation": [],
                    "delta_position": [],
                    "delta_euler": [],
                    "abs_position": [],
                    "abs_euler": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],  
                    "gripper_width": [],  
                },
            },
            "observation": {
                "is_image_encode": self.is_image_encode,
                "rgb": [],
                "rgb_timestamp": [],
            },
            "state": {
                "end_effector": {
                    "orientation": [],
                    "euler": [],
                    "position": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],
                    "gripper_width": [],
                },
            },
        }
        return data_dict

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()

    def get_data(self):
        return to_numpy(self.data_dict, self.device)

    def save_multi_cam_videos(self, rgb_array, base_path="videos", fps=30):
        """
        rgb_array shape: [frame_num, cam_num, H, W, 3]
        """
        os.makedirs(base_path, exist_ok=True)

        if isinstance(rgb_array, list):
            rgb_array = np.stack(rgb_array, axis=0)

        if rgb_array.ndim != 5 or rgb_array.shape[-1] != 3:
            raise ValueError("rgb_array must be shape [frame_num, cam_num, H, W, 3]")

        frame_num, cam_num, H, W, _ = rgb_array.shape

        for cam_idx in range(cam_num):
            video_filename = os.path.join(base_path, f"cam_{cam_idx}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'H264'
            out = cv2.VideoWriter(video_filename, fourcc, fps, (W, H))

            for frame_idx in range(frame_num):
                frame = rgb_array[frame_idx, cam_idx]  # [H, W, 3]
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"Saved camera {cam_idx} video to: {video_filename}")

    def save_data(self, save_path, episode_idx, is_compressed=False, is_save_video=True, is_save_hdf5=True, fps=30):
        """Save data as .npy file with dictionary structure, optionally also save as HDF5."""
        saving_data = to_numpy(self.data_dict, self.device)
        
        # Save as .npy or .npz
        # save_func = np.savez_compressed if is_compressed else np.save
        # np_path_data = os.path.join(save_path, f"data")
        # save_func(np_path_data, saving_data)
        # print(f"save data at {np_path_data}.{'npz' if is_compressed else 'npy'}.")
        
        # Save as HDF5
        if is_save_hdf5:
            hdf5_path = os.path.join(save_path, f"data")
            hdf5_saver = HDF5Saver(self.data_dict, self.device)
            hdf5_saver.save_to_hdf5(hdf5_path)
        
        # Save videos
        if is_save_video:
            if self.is_image_encode:
                raise ValueError(f"you set is_image_encode, so the video can not be saved.")
            self.save_multi_cam_videos(saving_data["observation"]["rgb"], save_path, fps=fps)
        
        self.clear_data()

    def update_rgb(self, timestamp=None):
        if self.cameras is None:
            return 
        rgb = self.cameras.get_rgb()
        timestamp = time.time() if timestamp==None else timestamp
        if self.is_image_encode:
            success, encoded_rgb = cv2.imencode('.jpeg', get_numpy(rgb, self.device), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("JPEG encode error.")
            rgb = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
        self.data_dict["observation"]["rgb"].append(rgb) # [N, camera_num, 480, 640, 3]
        self.data_dict["observation"]["rgb_timestamp"].append(timestamp)

    def update_state(self):
        joint_state = self.controller.get_joint_state()
        joint_pos_array = np.array(joint_state.pos()).copy()
        gripper_pos = joint_state.gripper_pos

        eef_state = self.controller.get_eef_state()
        pose_6d = np.array(eef_state.pose_6d()).copy()

        self.data_dict["state"]["joint"]["position"].append(joint_pos_array)
        self.data_dict["state"]["joint"]["gripper_width"].append(gripper_pos)
        self.data_dict["state"]["end_effector"]["position"].append(pose_6d[:3])
        self.data_dict["state"]["end_effector"]["euler"].append(pose_6d[3:])
        self.data_dict["state"]["end_effector"]["gripper_width"].append(gripper_pos)

    def update_action(self, save_action):
        action_joint = self.data_dict['action']["joint"]
        action_joint["position"].append(save_action["position"])
        action_joint["gripper_width"].append(save_action["gripper_width"])

        action_eef = self.data_dict['action']["end_effector"]
        action_eef["delta_position"].append(save_action["eef_delta_pos"])
        action_eef["delta_euler"].append(save_action["eef_delta_euler"])
        action_eef["abs_position"].append(save_action["eef_abs_pos"])
        action_eef["abs_euler"].append(save_action["eef_abs_euler"])
        action_eef["gripper_width"].append(save_action["gripper_width"])
        

    def update_data_dict(self, action, timestamp=None):
        self.update_rgb(timestamp)
        self.update_state()
        self.update_action(action)

class HDF5Saver:
    def __init__(self, data_dict, device='cpu'):
        self.data_dict = to_numpy(data_dict, device)

    def save_to_hdf5(self, save_path):
        """Save data as an HDF5 file."""
        with h5py.File(save_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            # Save action data
            action_group = root.create_group("action")
            for key, value in self.data_dict["action"]["joint"].items():
                action_group.create_dataset(f"joint/{key}", data=np.array(value))
            for key, value in self.data_dict["action"]["end_effector"].items():
                action_group.create_dataset(f"end_effector/{key}", data=np.array(value))

            # Save observation data
            observation_group = root.create_group("observation")
            observation_group.create_dataset("is_image_encode", data=self.data_dict["observation"]["is_image_encode"])
            observation_group.create_dataset("rgb_timestamp", data=np.array(self.data_dict["observation"]["rgb_timestamp"]))
            rgb_data = self.data_dict["observation"]["rgb"]
            has_rgb = False
            if isinstance(rgb_data, np.ndarray):
                has_rgb = rgb_data.size > 0
            else:
                try:
                    has_rgb = len(rgb_data) > 0
                except TypeError:
                    has_rgb = False

            if has_rgb:
                rgb_group = observation_group.create_group("rgb")
                for i, rgb in enumerate(rgb_data):
                    if isinstance(rgb, np.ndarray):
                        rgb_group.create_dataset(str(i), data=rgb)
                    else:
                        rgb_group.create_dataset(str(i), data=np.array(rgb, dtype=np.uint8))

            # Save state data
            state_group = root.create_group("state")
            for key, value in self.data_dict["state"]["joint"].items():
                state_group.create_dataset(f"joint/{key}", data=np.array(value))
            for key, value in self.data_dict["state"]["end_effector"].items():
                state_group.create_dataset(f"end_effector/{key}", data=np.array(value))

        print(f"Data saved to {save_path}.hdf5")
