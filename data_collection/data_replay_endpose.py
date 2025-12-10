"""
replay joint trajectory from hdf5 file
usage: python data_replay_endpose.py X5 can0 /media/le/data/pick_cube/pick_cube/episode_0/data.hdf5
"""

import os
import sys
import numpy as np
import h5py
import click
import time

# Adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Add path
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

# Import ARX-5 Interface
import arx5_interface as arx5

@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.argument("hdf5_path")
def main(model: str, interface: str, hdf5_path: str):
    controller = arx5.Arx5CartesianController(model, interface)
    controller.set_log_level(arx5.LogLevel.DEBUG)
    controller.reset_to_home()
    
    try:
        init_pose_6d = np.array([0.2398, 0.0012, 0.2185, 0.0039, 0.8967, 0.0035])
        init_gripper_pos = 0.0
        smooth_move_to_pose(controller, init_pose_6d, init_gripper_pos, duration=3.0)
        time.sleep(1.0)

        eef_waypoints = get_eef_waypoints(hdf5_path)

        # sending joint trajectory
        eef_traj = []
        init_timestamp = controller.get_eef_state().timestamp
        waypoint_interval_s = 0.05
        for waypoint in eef_waypoints:
            eef_traj.append(arx5.EEFState(waypoint[:6], waypoint[6], 0.0, 0.0))
            eef_traj[-1].timestamp = init_timestamp + waypoint_interval_s * len(eef_traj)
        
        controller.set_eef_traj(eef_traj)
        time.sleep(waypoint_interval_s * len(eef_traj))

        time.sleep(3.0)
        print("!!! Arrived at the end of the trajectory !!!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.reset_to_home()

def smooth_move_to_pose(controller: arx5.Arx5CartesianController, target_pose_6d, target_gripper_pos=0.0, duration=3.0):
    current_eef = controller.get_eef_state()
    current_pose_6d = np.array(current_eef.pose_6d()).copy()
    current_gripper_pos = current_eef.gripper_pos
    
    num_steps = int(duration / controller.get_controller_config().controller_dt)
    
    start_time = time.monotonic()
    
    for step in range(num_steps + 1):
        t = step / num_steps
        smooth_t = t * t * (3.0 - 2.0 * t)
        
        interpolated_pose = current_pose_6d + smooth_t * (target_pose_6d - current_pose_6d)
        interpolated_gripper = current_gripper_pos + smooth_t * (target_gripper_pos - current_gripper_pos)
        
        current_timestamp = controller.get_timestamp()
        eef_cmd = arx5.EEFState()
        eef_cmd.pose_6d()[:] = interpolated_pose
        eef_cmd.gripper_pos = interpolated_gripper
        eef_cmd.timestamp = current_timestamp + controller.get_controller_config().default_preview_time
        controller.set_eef_cmd(eef_cmd)
        
        target_time = start_time + (step + 1) * controller.get_controller_config().controller_dt
        while time.monotonic() < target_time:
            pass

    time.sleep(0.5)

def get_eef_waypoints(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        eef_pos = f['state/end_effector/position'][:]
        eef_euler = f['state/end_effector/euler'][:]
        gripper_width = f['state/end_effector/gripper_width'][:]

    eef_waypoints = np.concatenate([eef_pos, eef_euler, gripper_width[:, np.newaxis]], axis=1)

    return eef_waypoints

if __name__ == "__main__":
    main()