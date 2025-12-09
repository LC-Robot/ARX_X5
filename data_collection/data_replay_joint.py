"""
replay joint trajectory from hdf5 file
usage: python data_replay_joint.py X5 can0 /media/le/data/pick_cube/pick_cube/episode_0/data.hdf5
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
    controller = arx5.Arx5JointController(model, interface)
    controller.set_log_level(arx5.LogLevel.DEBUG)
    controller.reset_to_home()
    
    joint_waypoints = get_joint_waypoints(hdf5_path)

    # sending joint trajectory
    joint_traj = []
    init_timestamp = controller.get_joint_state().timestamp
    waypoint_interval_s = 0.5
    for waypoint in joint_waypoints:
        joint_traj.append(arx5.JointState(waypoint[:6], np.zeros(6), np.zeros(6), waypoint[6]))
        joint_traj[-1].timestamp = init_timestamp + waypoint_interval_s * len(joint_traj)
    
    controller.set_joint_traj(joint_traj)
    time.sleep(waypoint_interval_s * len(joint_traj))

    time.sleep(3.0)
    print("!!! Arrived at the end of the trajectory !!!")
    
    controller.reset_to_home()


def get_joint_waypoints(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        joint_pos = f['state/joint/position'][:]
        gripper_width = f['state/joint/gripper_width'][:]

    joint_waypoints = np.concatenate([joint_pos, gripper_width[:, np.newaxis]], axis=1)

    return joint_waypoints

if __name__ == "__main__":
    main()