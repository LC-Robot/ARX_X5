"""
SpaceMouse teleop data collection with multi-threaded camera capture to avoid control lag.
Example:
python data_collection.py --task_name pick --instruction "pick cube" --model X5 --interface can0 --save_hdf5 --save_video
"""
import os
import sys
import time
import json
import argparse
import threading
import numpy as np
from queue import Queue
from multiprocessing.managers import SharedMemoryManager
from pynput import keyboard
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel  
from peripherals.spacemouse_shared_memory import Spacemouse 
from data_collector import DataCollector  
from realsense_d435 import RealsenseAPI  


class CameraBuffer:
    """Thread-safe camera frame buffer"""

    def __init__(self):
        self.latest = None
        self.timestamp = None
        self.lock = threading.Lock()

    def set(self, frame, ts):
        with self.lock:
            self.latest = frame
            self.timestamp = ts

    def get(self):
        with self.lock:
            return self.latest, self.timestamp


def camera_worker(stop_event, cameras, buffer: CameraBuffer, fps: float = 15.0, show_video: bool = True):
    interval = 1.0 / fps if fps > 0 else 0.0
    last = time.monotonic()
    while not stop_event.is_set():
        now = time.monotonic()
        if interval > 0 and now - last < interval:
            time.sleep(max(0, interval - (now - last)))
        last = time.monotonic()
        try:
            rgb = cameras.get_rgb()  # [N, H, W, 3] or [H, W, 3]
            buffer.set(rgb, time.time())
            if show_video:
                if isinstance(rgb, np.ndarray) and rgb.ndim == 4:
                    frames_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in rgb]
                    display_frame = frames_bgr[0] if len(frames_bgr) == 1 else cv2.hconcat(frames_bgr)
                    cv2.imshow("Realsense Realtime", display_frame)
                else:
                    cv2.imshow("Realsense Realtime", rgb)
                cv2.waitKey(1)
        except Exception as e:
            print(f"[WARNING] Camera thread error: {e}")
            time.sleep(0.1)
    if show_video:
        cv2.destroyWindow("Realsense Realtime")


class SpacemouseDataCollection:
    def __init__(self, args, controller: Arx5CartesianController, cameras: RealsenseAPI, cam_buffer: CameraBuffer):
        self.args = args
        self.controller = controller
        self.cameras = cameras
        self.buffer = cam_buffer
        self.data_collector = DataCollector(controller, cameras)
        self.episode_idx = 0
        self.action_steps = 0
        self.instruction = args.instruction
        self.control_dt = 0.05
        self.preview_time = 0.05
        self.pos_speed = 0.4
        self.ori_speed = 0.8
        self.gripper_speed = 0.04
        self.is_collecting = False
        self.UPDATE_TRAJ = True

    def collect(self):
        print("[INFO] Use SpaceMouse to collect. Space: start/pause, Esc: quit without save, q: save and quit")
        should_exit = False
        save_flag = False

        # initialize state
        current_eef = self.controller.get_eef_state()
        target_pose_6d = np.array(current_eef.pose_6d()).copy()
        target_gripper_pos = current_eef.gripper_pos

        keyboard_state = {keyboard.Key.space: False, keyboard.Key.esc: False, keyboard.KeyCode.from_char("q"): False}
        def on_press(k):
            if k in keyboard_state:
                keyboard_state[k] = True
        def on_release(k):
            if k in keyboard_state:
                keyboard_state[k] = False
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=0.1, max_value=500) as sm:
                def filtered(sm):
                    state = sm.get_motion_state_transformed()
                    dz = 0.1
                    pos = state >= dz
                    neg = state <= -dz
                    state[pos] = (state[pos] - dz) / (1 - dz)
                    state[neg] = (state[neg] + dz) / (1 - dz)
                    return state

                start_time = time.monotonic()
                loop_cnt = 0
                while not should_exit:
                    # keyboard controls for state/exit
                    if keyboard_state[keyboard.Key.space]:
                        self.is_collecting = not self.is_collecting
                        print(f"[INFO] collect status: {'start' if self.is_collecting else 'pause'}")
                        time.sleep(0.3)
                    if keyboard_state[keyboard.Key.esc]:
                        should_exit = True
                        save_flag = False
                        print("[INFO] Exit without saving")
                    if keyboard_state[keyboard.KeyCode.from_char("q")]:
                        should_exit = True
                        save_flag = True
                        print("[INFO] Save and exit")

                    # spacemouse input
                    state = filtered(sm)
                    btn_l = sm.is_button_pressed(0)
                    btn_r = sm.is_button_pressed(1)
                    if btn_l and btn_r:
                        print("[INFO] Reset to home")
                        self.controller.reset_to_home()
                        target_pose_6d = self.controller.get_home_pose()
                        target_gripper_pos = 0.0
                        loop_cnt = 0
                        start_time = time.monotonic()
                        continue
                    elif btn_l and not btn_r:
                        gripper_cmd = 1
                    elif btn_r and not btn_l:
                        gripper_cmd = -1
                    else:
                        gripper_cmd = 0

                    target_pose_6d[:3] += state[:3] * self.pos_speed * self.control_dt
                    target_pose_6d[3:] += state[3:] * self.ori_speed * self.control_dt
                    target_gripper_pos += gripper_cmd * self.gripper_speed * self.control_dt
                    robot_cfg = self.controller.get_robot_config()
                    target_gripper_pos = np.clip(target_gripper_pos, 0, robot_cfg.gripper_width)

                    # send control
                    loop_cnt += 1
                    while time.monotonic() < start_time + loop_cnt * self.control_dt:
                        pass
                    ts = self.controller.get_timestamp()
                    eef_cmd = EEFState()
                    eef_cmd.pose_6d()[:] = target_pose_6d
                    eef_cmd.gripper_pos = target_gripper_pos
                    eef_cmd.timestamp = ts + self.preview_time

                    if self.UPDATE_TRAJ:
                        self.controller.set_eef_traj([eef_cmd])
                    else:
                        self.controller.set_eef_cmd(eef_cmd)

                    # record data
                    if self.is_collecting:
                        frame, frame_ts = self.buffer.get()
                        act = {
                            "position": np.array(self.controller.get_joint_state().pos()).copy(),
                            "gripper_width": target_gripper_pos,
                            "eef_delta_pos": state[:3] * self.pos_speed * self.control_dt,
                            "eef_delta_euler": state[3:] * self.ori_speed * self.control_dt,
                            "eef_abs_pos": target_pose_6d[:3].copy(),
                            "eef_abs_euler": target_pose_6d[3:].copy(),
                        }
                        self.data_collector.update_data_dict(action=act, timestamp=frame_ts, rgb_frame=frame)
                        self.action_steps += 1
                        if self.action_steps % 100 == 0:
                            print(f"[INFO] Collected {self.action_steps} steps")

        listener.stop()
        listener.join(timeout=1.0)
        return save_flag

    def save(self, task_dir):
        if self.action_steps > self.args.max_action_steps:
            print("action_steps too large, not saved")
            return False
        episode_idx = self.args.episode_idx if self.args.episode_idx >= 0 else self._next_episode_idx(task_dir)
        episode_dir = os.path.join(task_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)
        self.data_collector.save_data(
            episode_dir,
            episode_idx,
            is_save_video=not self.args.no_save_video,
            is_save_hdf5=self.args.save_hdf5,
            fps=self.args.cam_fps,
        )
        meta = {
            "task_name": self.args.task_name,
            "episode_idx": episode_idx,
            "action_steps": self.action_steps,
            "instruction": self.instruction,
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[INFO] Data saved to {episode_dir}")
        return True

    def _next_episode_idx(self, task_dir):
        if not os.path.exists(task_dir):
            return 0
        ids = []
        for name in os.listdir(task_dir):
            if name.startswith("episode_"):
                try:
                    ids.append(int(name.split("_")[1]))
                except Exception:
                    pass
        return (max(ids) + 1) if ids else 0

    def smooth_move_to_pose(self, target_pose_6d, target_gripper_pos=0.0, duration=3.0):
        current_eef = self.controller.get_eef_state()
        current_pose_6d = np.array(current_eef.pose_6d()).copy()
        current_gripper_pos = current_eef.gripper_pos
        
        num_steps = int(duration / self.control_dt)
        
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
            
            target_time = start_time + (step + 1) * self.control_dt
            while time.monotonic() < target_time:
                pass

        time.sleep(0.5)


def get_arguments():
    p = argparse.ArgumentParser(description="SpaceMouse data collection")
    p.add_argument("--dataset_dir", type=str, default="datasets", help="dataset directory")
    p.add_argument("--task_name", type=str, required=True, help="task name")
    p.add_argument("--instruction", type=str, required=True, help="task instruction/description")
    p.add_argument("--model", type=str, default="X5", help="robot model")
    p.add_argument("--interface", type=str, default="can0", help="CAN interface")
    p.add_argument("--min_action_steps", type=int, default=50)
    p.add_argument("--max_action_steps", type=int, default=2000)
    p.add_argument("--episode_idx", type=int, default=-1)
    p.add_argument("--save_hdf5", action="store_true", default=True)
    p.add_argument("--no_save_video", action="store_true")
    p.add_argument("--cam_fps", type=float, default=20.0, help="camera thread fps")
    return p.parse_args()


def main():
    args = get_arguments()
    print(f"[INFO] Init arm {args.model} @ {args.interface}")
    controller = Arx5CartesianController(args.model, args.interface)
    controller.set_log_level(LogLevel.INFO)
    controller.reset_to_home()
    time.sleep(1.0)

    # camera and thread
    cameras = RealsenseAPI()
    cam_buffer = CameraBuffer()
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=camera_worker,
        args=(stop_event, cameras, cam_buffer, args.cam_fps, True),
        daemon=True,
    )
    cam_thread.start()

    collector = SpacemouseDataCollection(args, controller, cameras, cam_buffer)

    # move to a better observation pose
    init_pose_6d = np.array([0.2398, 0.0012, 0.2185, 0.0039, 0.8967, 0.0035])
    init_gripper_pos = 0.0
    collector.smooth_move_to_pose(init_pose_6d, init_gripper_pos, duration=3.0)
    time.sleep(1.0)
    
    save_flag = collector.collect()

    stop_event.set()
    cam_thread.join(timeout=2.0)

    if not save_flag:
        print("[INFO] User chose not to save, exit.")
        controller.reset_to_home()
        return

    if collector.action_steps < args.min_action_steps:
        print(f"[ERROR] Steps too few {collector.action_steps} < {args.min_action_steps}, not saved.")
        controller.reset_to_home()
        return

    task_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)
    ok = collector.save(task_dir)
    if ok:
        print(f"[INFO] Done, total steps {collector.action_steps}")
    controller.reset_to_home()


if __name__ == "__main__":
    main()