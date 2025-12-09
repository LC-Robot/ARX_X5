"""
usage: python data_collection.py --task_name <task_name> --instruction <instruction> --can_port <can_port> --arm_type <arm_type> --pos_delta <pos_delta> --rot_delta <rot_delta> --gripper_delta <gripper_delta> --save_hdf5 --no_save_video --no_show_images
example: python data_collection.py --task_name "pick_1" --instruction "pick cube" --can_port "can0" --arm_type 0 --pos_delta 0.2 --rot_delta 1.0 --gripper_delta 0.04 --save_hdf5 --no_save_video
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import cv2
from queue import Queue
from pynput import keyboard
# Adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Add path
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel, JointState
from realsense_d435 import RealsenseAPI
from data_collector import DataCollector

class RealDataCollection:
    def __init__(self, args, controller: Arx5CartesianController, cameras: RealsenseAPI):
        self.controller: Arx5CartesianController = controller
        self.cameras: RealsenseAPI = cameras

        self.args = args
        self.data_collector = DataCollector(controller, cameras)

        self.episode_idx = 0  # Default episode index
        self.action_steps = 0
        self.instruction = args.instruction
        
        # 控制参数
        self.xyzrpy = np.zeros(6)  # 累积的末端位姿变化
        self.gripper_pos = 0.0  # 夹爪位置
        self.control_frequency = 10  # Hz
        self.control_time_step = 1.0 / self.control_frequency
        self.init_time = time.time()
        self.window_size = 1
        self.keyboard_queue = Queue(self.window_size)
        self.robot_config = controller.get_robot_config()
        self.controller_config = controller.get_controller_config()
        self.preview_time = 0.1
        
        # 控制增量
        self.pos_delta = args.pos_delta  # 位置增量 (m)
        self.rot_delta = args.rot_delta  # 旋转增量 (rad)
        self.gripper_delta = args.gripper_delta  # 夹爪增量
        
        self.is_collecting = False  # 是否正在采集数据
        self.show_images = not args.no_show_images  # 是否显示图像
        self.image_window_name = "Camera Views"  # 图像窗口名称
    
    def smooth_move_to_pose(self, target_pose_6d, target_gripper_pos=0.0, duration=3.0):
        # 获取当前位姿
        current_eef = self.controller.get_eef_state()
        current_pose_6d = np.array(current_eef.pose_6d()).copy()
        current_gripper_pos = current_eef.gripper_pos
        
        # 计算需要的步数
        num_steps = int(duration * self.control_frequency)
        
        # 生成插值轨迹
        start_time = time.monotonic()
        
        for step in range(num_steps + 1):
            # 计算插值因子（使用平滑的 S 曲线）
            t = step / num_steps
            # 使用三次平滑插值 (smoothstep)
            smooth_t = t * t * (3.0 - 2.0 * t)
            
            # 插值位姿
            interpolated_pose = current_pose_6d + smooth_t * (target_pose_6d - current_pose_6d)
            interpolated_gripper = current_gripper_pos + smooth_t * (target_gripper_pos - current_gripper_pos)
            
            # 发送命令
            current_timestamp = self.controller.get_timestamp()
            eef_cmd = EEFState()
            eef_cmd.pose_6d()[:] = interpolated_pose
            eef_cmd.gripper_pos = interpolated_gripper
            eef_cmd.timestamp = current_timestamp + self.preview_time
            self.controller.set_eef_cmd(eef_cmd)
            
            # 等待下一个控制周期
            target_time = start_time + (step + 1) * self.control_time_step
            while time.monotonic() < target_time:
                pass

        time.sleep(0.5)  # 稳定等待

    def reset_xyzrpy(self):
        """重置累积的位姿变化"""
        self.xyzrpy = np.zeros(6)
    
    def visualize_cameras(self, rgb_images):
        """
        显示多个相机的图像
        Args:
            rgb_images: numpy array of shape [n_cams, height, width, 3]
        """
        if rgb_images is None or not self.show_images:
            return
        
        n_cams = rgb_images.shape[0]
        
        if n_cams == 0:
            return
        
        # 如果只有一个相机，直接显示
        if n_cams == 1:
            img_bgr = cv2.cvtColor(rgb_images[0], cv2.COLOR_RGB2BGR)
            # 添加状态信息
            status_text = "collecting" if self.is_collecting else "stop"
            color = (0, 255, 0) if self.is_collecting else (0, 165, 255)
            cv2.putText(img_bgr, f"Status: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img_bgr, f"Steps: {self.action_steps}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.image_window_name, img_bgr)
        # 处理窗口事件（1ms等待，不阻塞）
        cv2.waitKey(1)

    def keyboard_control_loop(self):
        """使用键盘控制机械臂并采集数据"""
        
        # 退出标志
        should_exit = False
        save_data = True
        
        # 按键状态字典
        key_pressed = {
            keyboard.KeyCode.from_char('w'): False,  # X前移
            keyboard.KeyCode.from_char('s'): False,  # X后移
            keyboard.KeyCode.from_char('a'): False,  # Y左移
            keyboard.KeyCode.from_char('d'): False,  # Y右移
            keyboard.Key.up: False,  # Z上移
            keyboard.Key.down: False,  # Z下移
            keyboard.KeyCode.from_char('m'): False,  # Roll+
            keyboard.KeyCode.from_char('n'): False,  # Roll-
            keyboard.KeyCode.from_char('l'): False,  # Pitch+
            keyboard.KeyCode.from_char('.'): False,  # Pitch-
            keyboard.KeyCode.from_char(','): False,  # Yaw+
            keyboard.KeyCode.from_char('/'): False,  # Yaw-
            keyboard.KeyCode.from_char('o'): False,  # 张开夹爪
            keyboard.KeyCode.from_char('c'): False,  # 闭合夹爪
            keyboard.Key.space: False,  # 开始/暂停采集
            keyboard.KeyCode.from_char('q'): False,  # 退出并保存
            keyboard.Key.esc: False,  # 退出不保存
            keyboard.KeyCode.from_char('r'): False,  # 回零
        }
        
        # 按键回调函数
        def on_press(key):
            if key in key_pressed:
                key_pressed[key] = True
        
        def on_release(key):
            if key in key_pressed:
                key_pressed[key] = False
        
        # 启动键盘监听器
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        # 创建图像显示窗口
        if self.show_images:
            cv2.namedWindow(self.image_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.image_window_name, 640, 480)

        def get_filtered_keyboard_output(key_pressed: dict):
            state = np.zeros(6, dtype=np.float64)
            if key_pressed[keyboard.KeyCode.from_char('w')]:
                state[0] = 1
            if key_pressed[keyboard.KeyCode.from_char('s')]:
                state[0] = -1
            if key_pressed[keyboard.KeyCode.from_char('a')]:
                state[1] = 1
            if key_pressed[keyboard.KeyCode.from_char('d')]:
                state[1] = -1
            if key_pressed[keyboard.Key.up]:
                state[2] = 1
            if key_pressed[keyboard.Key.down]:
                state[2] = -1
            if key_pressed[keyboard.KeyCode.from_char('m')]:
                state[3] = 1
            if key_pressed[keyboard.KeyCode.from_char('n')]:
                state[3] = -1
            if key_pressed[keyboard.KeyCode.from_char('l')]:
                state[4] = 1
            if key_pressed[keyboard.KeyCode.from_char('.')]:
                state[4] = -1
            if key_pressed[keyboard.KeyCode.from_char(',')]:
                state[5] = 1
            if key_pressed[keyboard.KeyCode.from_char('/')]:
                state[5] = -1

            if (
                self.keyboard_queue.maxsize > 0
                and self.keyboard_queue._qsize() == self.keyboard_queue.maxsize
            ):
                self.keyboard_queue._get()

            self.keyboard_queue.put(state)

            return np.mean(np.array(list(self.keyboard_queue.queue)), axis=0)
            
        print("\n" + "=" * 70)
        print("数据采集控制台")
        print("=" * 70)
        print(f"任务: {self.instruction}")
        print("\n控制按键:")
        print("  [空格] 开始/暂停采集 | [q] 退出并保存 | [Esc] 退出不保存")
        print("  [r] 回零位置")
        print("  [w/s] X前/后 | [a/d] Y左/右 | [↑/↓] Z上/下")
        print("  [m/n] Roll+/- | [l/.] Pitch+/- | [,//] Yaw+/-")
        print("  [o] 张开夹爪 | [c] 闭合夹爪")
        print("=" * 70 + "\n")
        
        start_time = time.monotonic()
        loop_cnt = 0
        last_space_press_time = 0
        last_q_press_time = 0
        last_esc_press_time = 0
        last_i_press_time = 0
        last_r_press_time = 0
        
        # 初始化目标位姿和夹爪位置（从当前位置开始）
        current_eef = self.controller.get_eef_state()
        target_pose_6d = np.array(current_eef.pose_6d()).copy()
        target_gripper_pos = current_eef.gripper_pos
        
        try:
            while not should_exit:
                current_time = time.monotonic()
                
                # 显示当前状态
                if loop_cnt % 50 == 0:
                    ee_pose = self.controller.get_eef_state()
                    joint_pos = self.controller.get_joint_state()
                    
                    status_text = "collecting" if self.is_collecting else "stop"
                    print(f"\r采集状态: {status_text} | 步数: {self.action_steps} | "
                          f"末端位置: [{ee_pose.pose_6d()[0]:.3f}, {ee_pose.pose_6d()[1]:.3f}, {ee_pose.pose_6d()[2]:.3f}] | "
                          f"夹爪: {joint_pos.gripper_pos:.3f}    ", end='', flush=True)
                
                # 处理按键
                # 空格键 - 开始/暂停采集
                if key_pressed[keyboard.Key.space] and current_time - last_space_press_time > 0.3:
                    self.is_collecting = not self.is_collecting
                    status = "start" if self.is_collecting else "stop"
                    print(f"\n[INFO] {status}数据采集")
                    last_space_press_time = current_time
                
                # q键 - 退出并保存
                if key_pressed[keyboard.KeyCode.from_char('q')] and current_time - last_q_press_time > 0.3:
                    print("\n[INFO] 退出并保存数据...")
                    should_exit = True
                    save_data = True
                    last_q_press_time = current_time
                
                # Esc键 - 退出不保存
                if key_pressed[keyboard.Key.esc] and current_time - last_esc_press_time > 0.3:
                    print("\n[INFO] 退出，不保存数据...")
                    should_exit = True
                    save_data = False
                    self.action_steps = 0
                    last_esc_press_time = current_time
                
                # r键 - 回零
                if key_pressed[keyboard.KeyCode.from_char('r')] and current_time - last_r_press_time > 0.5:
                    target_pose_6d = self.controller.get_home_pose()
                    target_gripper_pos = 0.0
                    self.controller.reset_to_home()
                    print("\n[INFO] 回到零位")
                    last_r_press_time = current_time
                    continue
                
                # 获取键盘输入状态
                state = get_filtered_keyboard_output(key_pressed)
                key_open = key_pressed[keyboard.KeyCode.from_char('o')]
                key_close = key_pressed[keyboard.KeyCode.from_char('c')]

                # 夹爪控制命令
                if key_open and not key_close:
                    gripper_cmd = 1
                elif key_close and not key_open:
                    gripper_cmd = -1
                else:
                    gripper_cmd = 0

                # 累积位姿变化
                delta_pos = state[:3] * self.pos_delta * self.control_time_step
                delta_rot = state[3:] * self.rot_delta * self.control_time_step
                target_pose_6d[:3] += delta_pos
                target_pose_6d[3:] += delta_rot
                target_gripper_pos += gripper_cmd * self.gripper_delta * self.control_time_step
                
                # 夹爪位置限制
                if target_gripper_pos >= self.robot_config.gripper_width:
                    target_gripper_pos = self.robot_config.gripper_width
                elif target_gripper_pos <= 0:
                    target_gripper_pos = 0
                
                loop_cnt += 1
                
                # 等待到指定时间
                while time.monotonic() < start_time + loop_cnt * self.control_time_step:
                    pass

                # 发送控制命令
                current_timestamp = self.controller.get_timestamp()
                eef_cmd = EEFState()
                eef_cmd.pose_6d()[:] = target_pose_6d
                eef_cmd.gripper_pos = target_gripper_pos
                eef_cmd.timestamp = current_timestamp + self.preview_time
                self.controller.set_eef_cmd(eef_cmd)
                
                # 获取并显示相机图像
                if self.show_images and loop_cnt % 3 == 0:  # 每3帧显示一次，减少计算开销
                    try:
                        rgb_images = self.cameras.get_rgb()
                        self.visualize_cameras(rgb_images)
                    except Exception as e:
                        print(f"\n[WARNING] 图像显示错误: {e}")

                # 如果正在采集数据，记录数据
                if self.is_collecting:
                    timestamp = time.time() - self.init_time
                    
                    # 获取当前关节位置作为 action
                    current_joint_state = self.controller.get_joint_state()
                    joint_pos_array = np.array(current_joint_state.pos()).copy()
                    
                    save_action = {
                        "position": joint_pos_array,  
                        "gripper_width": current_joint_state.gripper_pos,  

                        "eef_delta_pos": delta_pos,
                        "eef_delta_euler": delta_rot,

                        "eef_abs_pos": target_pose_6d[:3].copy(),
                        "eef_abs_euler": target_pose_6d[3:].copy(),
                    }
                    
                    # 收集数据
                    self.data_collector.update_data_dict(
                        action=save_action,
                        timestamp=timestamp,
                    )
                    
                    self.action_steps += 1
                    
                    if self.action_steps % 100 == 0:
                        print(f"\n[INFO] 已采集 {self.action_steps} 步")
        
        except KeyboardInterrupt:
            print("\n[INFO] 收到键盘中断信号")
            should_exit = True
            save_data = True
        
        except Exception as e:
            print(f"\n[ERROR] 发生错误: {e}")
            import traceback
            traceback.print_exc()
            should_exit = True
            save_data = True
        
        finally:
            self.controller.reset_to_home()

            # 停止键盘监听器
            listener.stop()
            listener.join(timeout=1.0)
            print("\n[INFO] 键盘监听器已停止")
            
            # 关闭图像显示窗口
            if self.show_images:
                cv2.destroyAllWindows()
                print("[INFO] 图像显示窗口已关闭")
        
        return save_data
                
    def collect_data(self):
        """启动键盘控制的数据采集"""
        print("[INFO] 开始数据采集...")
        print("[INFO] 请使用键盘控制机械臂")
        print("[INFO] 按空格键开始/暂停采集，按q键退出并保存")
        return self.keyboard_control_loop()

    def get_next_episode_idx(self, task_dir):
        """
        Find the next episode index by identifying the highest existing episode number.

        Args:
            task_dir (str): The directory containing episode folders

        Returns:
            int: The next episode index (highest existing index + 1)
        """
        if not os.path.exists(task_dir):
            return 0  # Start with episode 0 if task directory doesn't exist

        # Get all items in the task directory
        all_items = os.listdir(task_dir)

        # Find all episode directories
        episode_dirs = []
        for item in all_items:
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path) and item.startswith("episode_"):
                episode_dirs.append(item)

        if not episode_dirs:
            return 0  # Start with episode 0 if no episode directories exist

        # Extract the episode numbers
        episode_numbers = []
        for dir_name in episode_dirs:
            try:
                # Extract number after "episode_"
                episode_number = int(dir_name.split("_")[1])
                episode_numbers.append(episode_number)
            except (IndexError, ValueError):
                # Skip directories that don't match the expected format
                continue

        if not episode_numbers:
            return 0  # If no valid episode numbers found, start with 0

        # Return the next episode index (max + 1)
        return max(episode_numbers) + 1

    def save_data(self, task_dir, controller):
        if self.action_steps > self.args.max_action_steps:
            print("action_steps too large, data not saved")
            return False
        if self.args.episode_idx < 0:
            self.episode_idx = self.get_next_episode_idx(task_dir)
        else:
            self.episode_idx = self.args.episode_idx
        episode_dir = os.path.join(task_dir, f"episode_{self.episode_idx}")

        # Ensure save_dir exists
        os.makedirs(episode_dir, exist_ok=True)
        metadata_path = os.path.join(episode_dir, "metadata.json")
        
        # Save data (with HDF5 option)
        self.data_collector.save_data(
            episode_dir, 
            self.episode_idx,
            is_save_video=not self.args.no_save_video,
            is_save_hdf5=self.args.save_hdf5,
            fps = self.control_frequency
        )

        # Save metadata
        metadata = {
            "task_name": self.args.task_name,
            "episode_idx": self.episode_idx,
            "action_steps": self.action_steps,
            "instruction": self.instruction,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Data saved to {episode_dir}")
        print(f"[INFO] Metadata saved to {metadata_path}")

        controller.reset_to_home() 

        return True


def get_arguments():
    parser = argparse.ArgumentParser(description="ARX-X5机械臂数据采集脚本")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="数据集保存目录")
    parser.add_argument("--task_name", type=str, required=True, help="任务名称")
    parser.add_argument("--min_action_steps", type=int, default=50, help="最小采集步数")
    parser.add_argument("--max_action_steps", type=int, default=2000, help="最大采集步数")
    parser.add_argument("--episode_idx", type=int, default=-1, help="Episode索引 (-1为自动递增)")
    parser.add_argument("--instruction", type=str, required=True, help="任务描述")
    parser.add_argument("--can_port", type=str, default="can0", help="CAN端口")
    parser.add_argument("--arm_type", type=int, default=0, help="机械臂类型 (0/1/2)")
    parser.add_argument("--pos_delta", type=float, default=0.1, help="位置控制增量 (m)")
    parser.add_argument("--rot_delta", type=float, default=0.5, help="旋转控制增量 (rad)")
    parser.add_argument("--gripper_delta", type=float, default=0.05, help="夹爪控制增量")
    parser.add_argument("--save_hdf5", action="store_true", default=True, help="同时保存为 HDF5 格式")
    parser.add_argument("--no_save_video", action="store_true", help="不保存视频")
    parser.add_argument("--no_show_images", action="store_true", help="不显示实时图像窗口")
    
    return parser.parse_args()

def main():
    args = get_arguments()
    
    # 初始化机械臂控制器
    model_map = {0: "X5", 1: "L5", 2: "X7"}
    model = model_map.get(args.arm_type, "X5")
    
    print(f"[INFO] 初始化机械臂: {model}, CAN端口: {args.can_port}")
    controller = Arx5CartesianController(model, args.can_port)
    controller.set_log_level(LogLevel.INFO)
    
    # 初始化相机
    cameras = RealsenseAPI()
    
    # 初始化数据采集器
    collection = RealDataCollection(args, controller, cameras)
    
    # 机械臂回零
    print("[INFO] 机械臂回零...")
    controller.reset_to_home()
    time.sleep(2)

    # 平滑移动到初始位姿（适用于腕部相机）
    init_ee_pose = np.array([0.2398, 0.0012, 0.2185, 0.0039, 0.8967, 0.0035])
    init_gripper_pos = 0.0
    collection.smooth_move_to_pose(init_ee_pose, init_gripper_pos, duration=3.0)
    
    # 开始数据采集
    save_data_flag = collection.collect_data()
    
    # 检查是否需要保存数据
    if not save_data_flag:
        print("[INFO] 用户选择不保存数据，退出。")
        controller.reset_to_home()
        exit(0)
    
    # 检查采集步数
    if collection.action_steps < args.min_action_steps:
        print(f"\033[31m[错误] 采集步数不足 ({collection.action_steps} < {args.min_action_steps})，数据未保存。\033[0m")
        controller.reset_to_home()
        exit(-1)
    
    if collection.action_steps == 0:
        print("[INFO] 未采集数据，退出。")
        controller.reset_to_home()
        exit(0)
    
    # 保存数据
    task_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    result = collection.save_data(task_dir, controller)
    if result:
        print(f"\033[32m\n保存成功！共采集 {collection.action_steps} 步数据。\033[0m\n")
    else:
        print(f"\033[31m\n保存失败！\033[0m\n")
        


if __name__ == "__main__":
    main()
