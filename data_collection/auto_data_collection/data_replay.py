"""
replay joint trajectory from waypoint file with randomized interpolation
usage: python data_replay.py X5 can0 waypoint1.txt
"""

import os
import sys
import numpy as np
import click
import time
from scipy.interpolate import CubicSpline

# adjust paths to find your utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# add path
ARX5_SDK_PATH = os.path.join(ROOT_DIR, "../../arx5-sdk/python")
UTILS_PATH = os.path.join(ROOT_DIR, "../../utils")
sys.path.insert(0, os.path.abspath(ARX5_SDK_PATH))
sys.path.insert(0, os.path.abspath(UTILS_PATH))

# Import ARX-5 Interface
import arx5_interface as arx5

@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.argument("waypoint_file")  # path to waypoint file
@click.option("--randomness", default=0.05, help="randomness level (0.0-1.0)")
@click.option("--points-per-segment", default=10, help="interpolation points between waypoints")
@click.option("--visualize", is_flag=True, help="visualize trajectory before execution")
@click.option("--repeat", default=1, help="number of times to repeat the trajectory")
def main(model: str, interface: str, waypoint_file: str, randomness: float, points_per_segment: int, visualize: bool, repeat: int):
    # initialize controller
    controller = arx5.Arx5JointController(model, interface)
    controller.set_log_level(arx5.LogLevel.INFO)
    
    print(f"Resetting to home position...")
    controller.reset_to_home()
    time.sleep(2)
    
    # load waypoints from file
    print(f"Loading waypoints from {waypoint_file}...")
    waypoints = load_waypoints_from_txt(waypoint_file)
    print(f"Loaded {len(waypoints)} waypoints")
    
    # repeat execution multiple times
    for iteration in range(repeat):
        if repeat > 1:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{repeat}")
            print(f"{'='*60}\n")
        
        # generate interpolated trajectory with randomness
        print(f"Generating interpolated trajectory (randomness: {randomness})...")
        trajectory = interpolate_with_randomness(waypoints, points_per_segment, randomness)
        print(f"Generated trajectory with {len(trajectory)} points")
        
        # visualize if requested
        if visualize:
            visualize_trajectory(waypoints, trajectory)
            input("Press Enter to continue with execution...")
        
        # send joint trajectory
        print("Executing trajectory...")
        joint_traj = []
        init_timestamp = controller.get_timestamp()
        dt = 0.1  # time between trajectory points (seconds)
        
        for i, point in enumerate(trajectory):
            joint_state = arx5.JointState(point[:6], np.zeros(6), np.zeros(6), point[6])
            joint_state.timestamp = init_timestamp + dt * (i + 1)
            joint_traj.append(joint_state)
        
        controller.set_joint_traj(joint_traj)
        
        # wait for trajectory completion
        total_time = dt * len(joint_traj)
        time.sleep(total_time + 1.0)
        
        print(f"Trajectory completed!")
        
        # pause between iterations
        if iteration < repeat - 1:
            time.sleep(2.0)
            print("Resetting to home position...")
            controller.reset_to_home()
            time.sleep(2.0)
    
    print("\nAll iterations completed!")
    time.sleep(2.0)
    print("Resetting to home position...")
    controller.reset_to_home()


def load_waypoints_from_txt(filepath):
    """
    Load waypoints from txt file
    Format: joint0,joint1,joint2,joint3,joint4,joint5,gripper
    
    Args:
        filepath: path to waypoint txt file
        
    Returns:
        numpy array of shape [num_waypoints, 7] (6 joints + gripper)
    """
    waypoints = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                values = [float(x) for x in line.split(',')]
                if len(values) == 7:
                    waypoints.append(values)
    
    return np.array(waypoints)


def interpolate_with_randomness(waypoints, points_per_segment=10, randomness=0.05):
    """
    Interpolate between waypoints with added randomness
    
    Args:
        waypoints: numpy array of shape [num_waypoints, 7]
        points_per_segment: number of interpolation points between each pair of waypoints
        randomness: level of randomness to add (0.0 = no randomness, 1.0 = high randomness)
        
    Returns:
        interpolated trajectory with randomness, shape [total_points, 7]
    """
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        return waypoints
    
    # create time indices for waypoints
    t_waypoints = np.arange(num_waypoints)
    
    # create dense time indices for interpolation
    total_points = (num_waypoints - 1) * points_per_segment + 1
    t_dense = np.linspace(0, num_waypoints - 1, total_points)
    
    # interpolate each dimension (6 joints + gripper)
    trajectory = np.zeros((total_points, 7))
    
    for dim in range(7):
        # use cubic spline interpolation for smooth trajectories
        cs = CubicSpline(t_waypoints, waypoints[:, dim])
        trajectory[:, dim] = cs(t_dense)
        
        # add randomness (except for gripper in some cases)
        if randomness > 0:
            # calculate the range of values for this dimension
            value_range = np.ptp(waypoints[:, dim])  # peak to peak (max - min)
            
            # for gripper (dimension 6), use less randomness
            if dim == 6:
                noise_scale = randomness * value_range * 0.3
            else:
                noise_scale = randomness * value_range * 0.5
            
            # generate smooth random noise using low-frequency sine waves
            num_waves = np.random.randint(2, 5)
            noise = np.zeros(total_points)
            for _ in range(num_waves):
                frequency = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.3, 1.0) * noise_scale
                noise += amplitude * np.sin(2 * np.pi * frequency * t_dense / num_waypoints + phase)
            
            # apply noise but keep waypoints fixed
            trajectory[:, dim] += noise
            
            # ensure waypoints remain at their original positions
            for i, t_wp in enumerate(t_waypoints):
                idx = int(t_wp * points_per_segment)
                if idx < total_points:
                    trajectory[idx, dim] = waypoints[i, dim]
    
    return trajectory


def visualize_trajectory(waypoints, trajectory):
    """
    Visualize the interpolated trajectory compared to original waypoints
    
    Args:
        waypoints: original waypoints [num_waypoints, 7]
        trajectory: interpolated trajectory [total_points, 7]
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 10))
        fig.suptitle('Trajectory Visualization with Randomness', fontsize=14, fontweight='bold')
        
        joint_names = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Gripper']
        
        for dim in range(7):
            row = dim // 2
            col = dim % 2
            ax = axes[row, col] if dim < 6 else axes[3, 0]
            
            # plot interpolated trajectory
            ax.plot(trajectory[:, dim], 'b-', linewidth=1.5, label='Interpolated with randomness')
            
            # plot waypoints
            waypoint_indices = np.linspace(0, len(trajectory) - 1, len(waypoints)).astype(int)
            ax.plot(waypoint_indices, waypoints[:, dim], 'ro', markersize=8, label='Waypoints')
            
            ax.set_xlabel('Time step')
            ax.set_ylabel('Value (rad or m)')
            ax.set_title(joint_names[dim])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # hide the last unused subplot
        axes[3, 1].axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        
        print("\nVisualization displayed. Close the window or press Enter to continue...")
        
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()