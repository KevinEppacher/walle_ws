#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
import rosbag
import os
import roslib

# ------------------------- CONFIGURABLE PARAMETERS -------------------------
# ROSBAG file paths
bag_nmpc_filename = 'recorded_data_nMPC_15.bag'
bag_dwa_filename = 'recorded_data_DWA_10.bag'
bag_teb_filename = 'recorded_data_TEB_2.bag'

# ROS topics for cmd_vel (used by different planners)
cmd_vel_topic = '/cmd_vel'

# ------------------------- FUNCTIONS -------------------------
def calculate_planner_frequency(bag_file_path, cmd_vel_topic):
    """Calculate the frequency of cmd_vel messages for a planner"""
    bag = rosbag.Bag(bag_file_path)
    
    # Count the number of cmd_vel messages and get the start/end times
    cmd_vel_count = 0
    start_time = None
    end_time = None

    for topic, msg, t in bag.read_messages(topics=[cmd_vel_topic]):
        if start_time is None:
            start_time = t  # Record the time of the first message
        end_time = t  # Update the end time with each message
        cmd_vel_count += 1

    # Calculate total duration
    if start_time and end_time:
        total_time = (end_time - start_time).to_sec()  # Convert from ROS time to seconds
        frequency = cmd_vel_count / total_time if total_time > 0 else 0
        return cmd_vel_count, total_time, frequency
    else:
        return 0, 0, 0  # If no messages were found

# Process path data from ROSBAG
def process_path_data(path_msg):
    path_x = []
    path_y = []
    for pose in path_msg.poses:
        path_x.append(pose.pose.position.x)
        path_y.append(pose.pose.position.y)
    return path_x, path_y

# Function to draw rotated rectangles (used for cubes and other shapes)
def draw_rotated_rectangle(x, y, width, height, angle, ax):
    """Draw a rotated rectangle."""
    theta = np.radians(angle)
    rect = patches.Rectangle((-width/2, -height/2), width, height, linewidth=1, edgecolor='k', facecolor='k')
    t = plt.gca().transData
    t_rot = Affine2D().rotate(theta).translate(x, y) + t
    rect.set_transform(t_rot)
    ax.add_patch(rect)

# ------------------------- MAIN PROGRAM -------------------------
def main():
    # Path to the ROSBAG directory
    amr_control_path = roslib.packages.get_pkg_dir('amr_control')

    # Full paths to the bags
    bag_nmpc_path = os.path.join(amr_control_path, f'data/rosbags/{bag_nmpc_filename}')
    bag_dwa_path = os.path.join(amr_control_path, f'data/rosbags/{bag_dwa_filename}')
    bag_teb_path = os.path.join(amr_control_path, f'data/rosbags/{bag_teb_filename}')

    # Calculate cmd_vel frequencies for each planner
    nmpc_cmd_vel_count, nmpc_total_time, nmpc_frequency = calculate_planner_frequency(bag_nmpc_path, cmd_vel_topic)
    dwa_cmd_vel_count, dwa_total_time, dwa_frequency = calculate_planner_frequency(bag_dwa_path, cmd_vel_topic)
    teb_cmd_vel_count, teb_total_time, teb_frequency = calculate_planner_frequency(bag_teb_path, cmd_vel_topic)

    # Print results to the terminal
    print(f"nMPC Planner: {nmpc_cmd_vel_count} messages, {nmpc_total_time:.2f} seconds, {nmpc_frequency:.2f} Hz")
    print(f"DWA Planner: {dwa_cmd_vel_count} messages, {dwa_total_time:.2f} seconds, {dwa_frequency:.2f} Hz")
    print(f"TEB Planner: {teb_cmd_vel_count} messages, {teb_total_time:.2f} seconds, {teb_frequency:.2f} Hz")

    # ------------------------- PLOTTING -------------------------
    # 1. Plot for the paths of nMPC, DWA, and TEB including objects (walls, cubes, cylinders)
    fig, ax1 = plt.subplots()

    # Drawing the environment (walls, cylinders, rectangles)
    walls = [
        (-14.925, 0, 5, 0.15, -90),
        (-0.031145, -2.425, 30, 0.15, 0),
        (-5.055, 1.375, 2.25, 0.15, -90),
        (-4.88, 0.325, 0.5, 0.15, 0),
        (-4.705, 1.375, 2.25, 0.15, 90),
        (-4.88, -0.525, 0.5, 0.15, 0),
        (-4.705, -1.45, 2, 0.15, -90),
        (-5.055, -1.45, 2, 0.15, -90),
        (14.925, 0, 5, 0.15, 90),
        (-0.031145, 2.425, 30, 0.15, 180),
        (5.185, -1.44, 2, 0.15, -90),
        (5.36, -0.515, 0.5, 0.15, 0),
        (5.535, -1.44, 2, 0.15, -90),
        (5.175, 1.365, 2.25, 0.15, 90),
        (5.35, 0.315, 0.5, 0.15, 0),
        (5.525, 1.365, 2.25, 0.15, 90)
    ]

    # Drawing walls as black-filled rectangles with rotation
    for wall in walls:
        draw_rotated_rectangle(wall[0], wall[1], wall[2], wall[3], wall[4], ax1)

    # Drawing cylinders
    cylinders = [
        (-2.664431, 1.662781,  0.278354, 0),
        (-3.11174, -1.27262, 0.278354, 0),
        (3.55449, -1.48911, 0.278354, 0),
        (-0.526371, -1.58176, 0.278354, 0),
        (-4.12027, 0.785226, 0.278354, 0)
    ]

    # Drawing black-filled circles (cylinders)
    for cylinder in cylinders:
        circ = patches.Circle((cylinder[0], cylinder[1]), cylinder[2], linewidth=1, edgecolor='k', facecolor='k')
        ax1.add_patch(circ)

    # Drawing the paths from the ROSBAG
    # Extracting the paths for nMPC, DWA, and TEB
    nmpc_path_x, nmpc_path_y = [], []
    dwa_path_x, dwa_path_y = [], []
    teb_path_x, teb_path_y = [], []

    # Reading nMPC path
    bag_nmpc = rosbag.Bag(bag_nmpc_path)
    for topic, msg, t in bag_nmpc.read_messages(topics=['/robot_path_DWA']):
        nmpc_path_x, nmpc_path_y = process_path_data(msg)
    bag_nmpc.close()

    # Reading DWA path
    bag_dwa = rosbag.Bag(bag_dwa_path)
    for topic, msg, t in bag_dwa.read_messages(topics=['/robot_path_DWA']):
        dwa_path_x, dwa_path_y = process_path_data(msg)
    bag_dwa.close()

    # Reading TEB path
    bag_teb = rosbag.Bag(bag_teb_path)
    for topic, msg, t in bag_teb.read_messages(topics=['/robot_path_DWA']):
        teb_path_x, teb_path_y = process_path_data(msg)
    bag_teb.close()

    # Plotting the paths
    ax1.plot(nmpc_path_x, nmpc_path_y, label='nMPC Path', color='green', linestyle='-', linewidth=2)
    ax1.plot(dwa_path_x, dwa_path_y, label='DWA Path', color='red', linestyle='--', linewidth=2)
    ax1.plot(teb_path_x, teb_path_y, label='TEB Path', color='blue', linestyle='-.', linewidth=2)

    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_title('Paths for nMPC, DWA, and TEB with Objects', fontsize=14)
    ax1.legend()

    # Show the plot for the paths
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # 2. Plot for Computation Time
    fig, ax2 = plt.subplots()
    planners = ['nMPC', 'DWA', 'TEB']
    computation_times = [nmpc_total_time, dwa_total_time, teb_total_time]
    
    ax2.bar(planners, computation_times, color=['green', 'red', 'blue'])
    ax2.set_xlabel('Planners', fontsize=12)
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax2.set_title('Computation Time Comparison', fontsize=14)

    # Show the plot for computation time
    plt.grid(True)
    plt.show()

    # 3. Plot for Arrival Time
    fig, ax3 = plt.subplots()
    arrival_times = [nmpc_total_time, dwa_total_time, teb_total_time]  # Here, assuming the same as computation time
    
    ax3.bar(planners, arrival_times, color=['green', 'red', 'blue'])
    ax3.set_xlabel('Planners', fontsize=12)
    ax3.set_ylabel('Arrival Time (seconds)', fontsize=12)
    ax3.set_title('Arrival Time Comparison', fontsize=14)

    # Show the plot for arrival time
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
