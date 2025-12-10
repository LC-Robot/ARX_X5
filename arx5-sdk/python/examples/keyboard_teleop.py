import time
from pynput import keyboard
from queue import Queue
import os
import sys
import numpy as np
import cv2  # [新增] 导入 OpenCV
import click

# Adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Add path
UTILS_PATH = os.path.join(ROOT_DIR, "../../../utils")
sys.path.insert(0, os.path.abspath(UTILS_PATH))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR) # 建议注释掉这行，改变工作目录有时会影响文件读取，除非必须

from realsense_d435 import RealsenseAPI
from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel
from multiprocessing.managers import SharedMemoryManager


def start_keyboard_teleop(controller: Arx5CartesianController, cameras: RealsenseAPI):

    ori_speed = 1.0
    pos_speed = 0.4
    gripper_speed = 0.04
    
    # initialize target pose from current pose instead of home pose
    current_eef = controller.get_eef_state()
    target_pose_6d = np.array(current_eef.pose_6d()).copy()
    target_gripper_pos = current_eef.gripper_pos
    cmd_dt = 0.01
    preview_time = 0.1
    window_size = 5
    keyboard_queue = Queue(window_size)
    robot_config = controller.get_robot_config()
    controller_config = controller.get_controller_config()

    print("Teleop tracking started. Press 'q' in terminal to exit (or close window).")

    key_pressed = {
        keyboard.Key.up: False,  # +x
        keyboard.Key.down: False,  # -x
        keyboard.Key.left: False,  # +y
        keyboard.Key.right: False,  # -y
        keyboard.Key.page_up: False,  # +z
        keyboard.Key.page_down: False,  # -z
        keyboard.KeyCode.from_char("q"): False,  # +roll
        keyboard.KeyCode.from_char("a"): False,  # -roll
        keyboard.KeyCode.from_char("w"): False,  # +pitch
        keyboard.KeyCode.from_char("s"): False,  # -pitch
        keyboard.KeyCode.from_char("e"): False,  # +yaw
        keyboard.KeyCode.from_char("d"): False,  # -yaw
        keyboard.KeyCode.from_char("r"): False,  # open gripper
        keyboard.KeyCode.from_char("f"): False,  # close gripper
        keyboard.Key.space: False,  # reset to home
        keyboard.Key.esc: False,    # ESC to exit
        keyboard.KeyCode.from_char("p"): False,  # save waypoint
    }

    def on_press(key):
        if key in key_pressed:
            key_pressed[key] = True
        # 允许通过 ESC 键退出
        if key == keyboard.Key.esc:
            key_pressed[keyboard.Key.esc] = True

    def on_release(key):
        if key in key_pressed:
            key_pressed[key] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def get_filtered_keyboard_output(key_pressed: dict):
        state = np.zeros(6, dtype=np.float64)
        if key_pressed[keyboard.Key.up]:
            state[0] = 1
        if key_pressed[keyboard.Key.down]:
            state[0] = -1
        if key_pressed[keyboard.Key.left]:
            state[1] = 1
        if key_pressed[keyboard.Key.right]:
            state[1] = -1
        if key_pressed[keyboard.Key.page_up]:
            state[2] = 1
        if key_pressed[keyboard.Key.page_down]:
            state[2] = -1
        if key_pressed[keyboard.KeyCode.from_char("q")]:
            state[3] = 1
        if key_pressed[keyboard.KeyCode.from_char("a")]:
            state[3] = -1
        if key_pressed[keyboard.KeyCode.from_char("w")]:
            state[4] = 1
        if key_pressed[keyboard.KeyCode.from_char("s")]:
            state[4] = -1
        if key_pressed[keyboard.KeyCode.from_char("e")]:
            state[5] = 1
        if key_pressed[keyboard.KeyCode.from_char("d")]:
            state[5] = -1

        if (
            keyboard_queue.maxsize > 0
            and keyboard_queue._qsize() == keyboard_queue.maxsize
        ):
            keyboard_queue._get()

        keyboard_queue.put(state)

        return np.mean(np.array(list(keyboard_queue.queue)), axis=0)

    directions = np.zeros(6, dtype=np.float64)
    start_time = time.monotonic()
    loop_cnt = 0
    last_p_press_time = 0  # debounce for waypoint saving
    waypoint_file = "/home/le/ARX_X5/data_collection/auto_data_collection/waypoint1.txt"
    
    # ensure waypoint directory exists
    os.makedirs(os.path.dirname(waypoint_file), exist_ok=True)
    
    try:
        while True:
            # 1. 检查退出条件
            if key_pressed[keyboard.Key.esc]:
                print("Exiting...")
                break

            # 2. 获取并显示相机画面
            # RealsenseAPI 通常返回 [N, H, W, 3] 或 [H, W, 3] 的 RGB 数组
            rgb_images = cameras.get_rgb()
            
            # 处理多相机情况，这里默认取第一个相机
            if isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4:
                vis_img = rgb_images[0]
            else:
                vis_img = rgb_images

            # OpenCV 使用 BGR 格式，需要转换
            if vis_img is not None:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Realsense Realtime", vis_img)
                # waitKey 是必须的，用于刷新 GUI 事件，1ms 延迟
                if cv2.waitKey(1) & 0xFF == 27: # 也可以按 ESC 退出窗口
                    break

            # 3. 获取机器人状态并打印
            eef_state = controller.get_eef_state()
            # joint_state = controller.get_joint_state()
            print(
                f"Time: {time.monotonic() - start_time:.02f}s | POS: {eef_state.pose_6d()[:3]}",
                end="\r",
            )

            # 4. 处理键盘输入控制
            prev_directions = directions
            directions = np.zeros(7, dtype=np.float64)
            state = get_filtered_keyboard_output(key_pressed)
            key_open = key_pressed[keyboard.KeyCode.from_char("r")]
            key_close = key_pressed[keyboard.KeyCode.from_char("f")]
            key_space = key_pressed[keyboard.Key.space]
            key_p = key_pressed[keyboard.KeyCode.from_char("p")]
            
            # save current joint state to waypoint file
            current_time = time.monotonic()
            if key_p and (current_time - last_p_press_time > 0.5):
                joint_state = controller.get_joint_state()
                joint_pos = joint_state.pos()
                gripper_pos = joint_state.gripper_pos
                
                # append waypoint to file
                with open(waypoint_file, 'a') as f:
                    waypoint_str = f"{joint_pos[0]:.6f},{joint_pos[1]:.6f},{joint_pos[2]:.6f},{joint_pos[3]:.6f},{joint_pos[4]:.6f},{joint_pos[5]:.6f},{gripper_pos:.6f}\n"
                    f.write(waypoint_str)
                
                last_p_press_time = current_time

            if key_space:
                controller.reset_to_home()
                target_pose_6d = controller.get_home_pose()
                target_gripper_pos = 0.0
                loop_cnt = 0
                start_time = time.monotonic()
                continue
            elif key_open and not key_close:
                gripper_cmd = 1
            elif key_close and not key_open:
                gripper_cmd = -1
            else:
                gripper_cmd = 0

            target_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
            target_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
            target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
            
            if target_gripper_pos >= robot_config.gripper_width:
                target_gripper_pos = robot_config.gripper_width
            elif target_gripper_pos <= 0:
                target_gripper_pos = 0
            
            # 5. 频率控制
            loop_cnt += 1
            while time.monotonic() < start_time + loop_cnt * cmd_dt:
                pass

            # 6. 发送指令
            current_timestamp = controller.get_timestamp()
            eef_cmd = EEFState()
            eef_cmd.pose_6d()[:] = target_pose_6d
            eef_cmd.gripper_pos = target_gripper_pos
            eef_cmd.timestamp = current_timestamp + preview_time
            controller.set_eef_cmd(eef_cmd)

            # joint_state = controller.get_joint_state()
            # print(f"joint state: {joint_state.pos()}")
            eef_state = controller.get_eef_state()
            print(f"eef state: {eef_state.pose_6d()}")

    finally:
        # 停止监听和销毁窗口
        listener.stop()
        cv2.destroyAllWindows()

def smooth_move_to_pose(controller: Arx5CartesianController, target_pose_6d, target_gripper_pos=0.0, duration=3.0):
    print(f"Smoothly moving to target pose: {target_pose_6d}")
    
    # get current pose
    current_eef = controller.get_eef_state()
    current_pose_6d = np.array(current_eef.pose_6d()).copy()
    current_gripper_pos = current_eef.gripper_pos
    
    # get controller configuration
    controller_config = controller.get_controller_config()
    control_dt = controller_config.controller_dt
    preview_time = controller_config.default_preview_time
    control_frequency = 1.0 / control_dt
    
    # calculate number of steps
    num_steps = int(duration * control_frequency)
    
    # generate interpolated trajectory
    start_time = time.monotonic()
    
    for step in range(num_steps + 1):
        # calculate interpolation factor using smooth S-curve
        t = step / num_steps
        # use cubic smoothstep interpolation
        smooth_t = t * t * (3.0 - 2.0 * t)
        
        # interpolate pose
        interpolated_pose = current_pose_6d + smooth_t * (target_pose_6d - current_pose_6d)
        interpolated_gripper = current_gripper_pos + smooth_t * (target_gripper_pos - current_gripper_pos)
        
        # send command
        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = interpolated_pose
        eef_cmd.gripper_pos = interpolated_gripper
        eef_cmd.timestamp = current_timestamp + preview_time
        controller.set_eef_cmd(eef_cmd)

        
        # wait for next control cycle
        target_time = start_time + (step + 1) * control_dt
        while time.monotonic() < target_time:
            pass
    
    print("\nMovement completed!")
    time.sleep(0.5)  # stabilization wait


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    # initialize controller
    controller = Arx5CartesianController(model, interface)
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)
    
    # reset to home position
    print("Resetting to home position...")
    controller.reset_to_home()
    time.sleep(2)  # wait for stabilization

    # smoothly move to initial pose for wrist camera
    init_ee_pose = np.array([0.2398, 0.0012, 0.2185, 0.0039, 0.8967, 0.0035])
    init_gripper_pos = 0.0
    smooth_move_to_pose(controller, init_ee_pose, init_gripper_pos, duration=3.0)

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    
    # initialize RealSense camera
    print("\nInitializing RealSense Camera...")
    try:
        cameras = RealsenseAPI()
        print("Camera initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    # start keyboard teleoperation
    try:
        start_keyboard_teleop(controller, cameras)
    except KeyboardInterrupt:
        print(f"\nTeleop recording is terminated. Resetting to home.")
    finally:
        controller.reset_to_home()
        # controller.set_to_damping()  # optional, enable if needed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()