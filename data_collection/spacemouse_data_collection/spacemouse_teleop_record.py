from queue import Queue
import os
import sys
import cv2
import numpy as np
import time
import click
import threading
from multiprocessing.managers import SharedMemoryManager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# add utils path for Realsense
UTILS_PATH = os.path.abspath(os.path.join(ROOT_DIR, "../../../utils"))
sys.path.insert(0, UTILS_PATH)

os.chdir(ROOT_DIR)
from arx5_interface import (
    Arx5CartesianController,
    ControllerConfig,
    ControllerConfigFactory,
    EEFState,
    Gain,
    LogLevel,
    RobotConfigFactory,
)
from peripherals.spacemouse_shared_memory import Spacemouse
from realsense_d435 import RealsenseAPI


def start_teleop_recording(controller: Arx5CartesianController):

    ori_speed = 0.8
    pos_speed = 0.4
    gripper_speed = 0.04
    # For earlier spacemouse versions (wired version), the readout might be not zero even after it is released
    # If you are using the wireless 3Dconnexion spacemouse, you can set the deadzone_threshold to 0.0 for better sensitivity
    deadzone_threshold = 0.1
    target_pose_6d = controller.get_home_pose()

    target_gripper_pos = 0.0

    window_size = 3
    cmd_dt = 0.01
    preview_time = 0.05  # Each trajectory command is 0.15s ahead of the current time

    UPDATE_TRAJ = True
    # False: only override single points with position control
    # True: send a trajectory command every update_dt. Will include velocity. This is

    pose_x_min = target_pose_6d[0]
    spacemouse_queue = Queue(window_size)
    robot_config = controller.get_robot_config()

    avg_error = np.zeros(6)
    avg_cnt = 0
    prev_eef_cmd = EEFState()
    eef_cmd = EEFState()

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(
            shm_manager=shm_manager, deadzone=deadzone_threshold, max_value=500
        ) as sm:

            def get_filtered_spacemouse_output(sm: Spacemouse):
                state = sm.get_motion_state_transformed()
                # Remove the deadzone and normalize the output
                positive_idx = state >= deadzone_threshold
                negative_idx = state <= -deadzone_threshold
                state[positive_idx] = (state[positive_idx] - deadzone_threshold) / (
                    1 - deadzone_threshold
                )
                state[negative_idx] = (state[negative_idx] + deadzone_threshold) / (
                    1 - deadzone_threshold
                )

                if (
                    spacemouse_queue.maxsize > 0
                    and spacemouse_queue._qsize() == spacemouse_queue.maxsize
                ):
                    spacemouse_queue._get()
                spacemouse_queue.put_nowait(state)
                return np.mean(np.array(list(spacemouse_queue.queue)), axis=0)

            print("Teleop tracking ready. Waiting for spacemouse movement to start.")

            while True:
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                state = get_filtered_spacemouse_output(sm)
                if state.any() or button_left or button_right:
                    print(f"Start tracking!")
                    break
                eef_cmd = controller.get_eef_cmd()
                prev_eef_cmd = eef_cmd
            start_time = time.monotonic()
            loop_cnt = 0
            while True:

                print(
                    f"Time elapsed: {time.monotonic() - start_time:.03f}s",
                    end="\r",
                )
                # Spacemouse state is in the format of (x y z roll pitch yaw)
                state = get_filtered_spacemouse_output(sm)
                button_left = sm.is_button_pressed(0)
                button_right = sm.is_button_pressed(1)
                if button_left and button_right:
                    print(f"Avg 6D pose error: {avg_error / avg_cnt}")
                    # Traj with vel Avg 6D pose error:      [ 0.0004  0.0002 -0.0016  0.0002  0.0032  0.0005]
                    # Single point without vel:             [-0.0002 -0.0006 -0.0026  0.0027  0.0042 -0.0017]
                    # Traj without vel Avg 6D pose error:   [ 0.0005  0.0008 -0.005  -0.0024  0.0073 -0.0001]

                    controller.reset_to_home()
                    config = controller.get_robot_config()
                    target_pose_6d = controller.get_home_pose()
                    target_gripper_pos = 0.0
                    loop_cnt = 0
                    start_time = time.monotonic()

                    continue
                elif button_left and not button_right:
                    gripper_cmd = 1
                elif button_right and not button_left:
                    gripper_cmd = -1
                else:
                    gripper_cmd = 0
                # print(state, target_gripper_pos)
                target_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
                target_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
                target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
                if target_gripper_pos >= robot_config.gripper_width:
                    target_gripper_pos = robot_config.gripper_width
                elif target_gripper_pos <= 0:
                    target_gripper_pos = 0

                loop_cnt += 1
                while time.monotonic() < start_time + loop_cnt * cmd_dt:
                    pass
                current_timestamp = controller.get_timestamp()
                prev_eef_cmd = eef_cmd
                # if target_pose_6d[0] < pose_x_min:
                #     target_pose_6d[0] = pose_x_min
                eef_cmd.pose_6d()[:] = target_pose_6d
                eef_cmd.gripper_pos = target_gripper_pos
                eef_cmd.timestamp = current_timestamp + preview_time

                if UPDATE_TRAJ:
                    # This will calculate the velocity automatically
                    controller.set_eef_traj([eef_cmd])

                # Or sending single eef_cmd:
                else:
                    # Only position control
                    controller.set_eef_cmd(eef_cmd)

                output_eef_cmd = controller.get_eef_cmd()
                eef_state = controller.get_eef_state()
                avg_error += output_eef_cmd.pose_6d() - eef_state.pose_6d()
                avg_cnt += 1

                print(f"6DPose Error: {output_eef_cmd.pose_6d() - eef_state.pose_6d()}")

    # no camera resources allocated in control loop
    pass


def camera_worker(
    stop_event: threading.Event,
    cameras: RealsenseAPI,
    save_video: bool,
    video_dir: str,
    show_video: bool,
    fps: float = 15.0,
):
    """
    Separate thread to grab frames and optionally record/display.
    Keeps control loop light.
    """
    video_writers = []
    video_writers_ok = False

    video_dir = os.path.abspath(video_dir)
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        num_cams = cameras.get_num_cameras()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_idx in range(num_cams):
            video_path = os.path.join(video_dir, f"spacemouse_cam_{cam_idx}.mp4")
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                cameras.fps,
                (cameras.width, cameras.height),
            )
            video_writers.append(writer)
        video_writers_ok = all(w.isOpened() for w in video_writers)
        if video_writers_ok:
            print(f"[INFO] Video recording enabled. Saving to {video_dir}")
        else:
            print(f"[WARNING] VideoWriter failed to open. Video will not be saved.")

    interval = 1.0 / fps if fps > 0 else 0.0
    last_time = time.monotonic()

    while not stop_event.is_set():
        now = time.monotonic()
        if interval > 0 and now - last_time < interval:
            time.sleep(max(0.0, interval - (now - last_time)))
        last_time = time.monotonic()

        try:
            rgb_images = cameras.get_rgb()  # [N, H, W, 3] uint8
            if isinstance(rgb_images, np.ndarray) and rgb_images.ndim == 4:
                frames_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in rgb_images]
                display_frame = frames_bgr[0] if len(frames_bgr) == 1 else cv2.hconcat(frames_bgr)

                if save_video and video_writers_ok and len(video_writers) == len(frames_bgr):
                    for w, f in zip(video_writers, frames_bgr):
                        w.write(f)

                if show_video:
                    cv2.imshow("Realsense Realtime", display_frame)
                    cv2.waitKey(1)
            else:
                if show_video:
                    cv2.imshow("Realsense Realtime", rgb_images)
                    cv2.waitKey(1)
        except Exception as cam_err:
            print(f"[WARNING] Camera fetch/display failed: {cam_err}")
            time.sleep(0.1)

    # cleanup
    for writer in video_writers:
        writer.release()
    if show_video:
        cv2.destroyAllWindows()


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--save-video", is_flag=True, help="save camera video")
@click.option("--video-dir", default="videos", help="directory to save videos")
@click.option("--show-video", is_flag=True, help="show realtime camera window")
@click.option("--cam-fps", default=15.0, help="camera capture fps for recording")
def main(model: str, interface: str, save_video: bool, video_dir: str, show_video: bool, cam_fps: float):

    robot_config = RobotConfigFactory.get_instance().get_config(model)
    controller_config = ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config.joint_dof
    )
    # controller_config.interpolation_method = "cubic"
    controller_config.default_kp = controller_config.default_kp
    controller = Arx5CartesianController(robot_config, controller_config, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)

    # initialize cameras and start camera thread
    cameras = None
    cam_thread = None
    stop_event = threading.Event()
    if save_video or show_video:
        try:
            cameras = RealsenseAPI()
            cam_thread = threading.Thread(
                target=camera_worker,
                args=(stop_event, cameras, save_video, video_dir, show_video, cam_fps),
                daemon=True,
            )
            cam_thread.start()
        except Exception as cam_err:
            print(f"[WARNING] Failed to initialize Realsense camera: {cam_err}")

    try:
        start_teleop_recording(controller)
    except KeyboardInterrupt:
        print(f"Teleop recording is terminated. Resetting to home.")
    finally:
        stop_event.set()
        if cam_thread is not None:
            cam_thread.join(timeout=2.0)
        controller.reset_to_home()
        controller.set_to_damping()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
