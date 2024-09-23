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
# bag_nmpc_filename = 'recorded_data_nMPC_22.bag'
# bag_dwa_filename = 'recorded_data_DWA_10.bag'
# bag_teb_filename = 'recorded_data_TEB_8.bag'

bag_nmpc_filename = 'recorded_data_nMPC_23.bag'
bag_dwa_filename = 'recorded_data_DWA_12.bag'
bag_teb_filename = 'recorded_data_TEB_9.bag'

# bag_nmpc_filename = 'recorded_data_nMPC_18.bag'
# bag_dwa_filename = 'recorded_data_DWA_15.bag'
# bag_teb_filename = 'recorded_data_TEB_10.bag'


# ROS topics for cmd_vel (used by different planners)
cmd_vel_topic = '/cmd_vel'
global_plan_topic = '/move_base/NavfnROS/plan'

# Angle for cube placement along the dashed circle (in degrees)
cube_angle_on_circle_1 = 45  # Set the angle where you want to place the cube
cube_size = 0.2  # Size of the cube (0.2 x 0.2)

# Angle for cube placement along the dashed circle (in degrees)
cube_angle_on_circle_2 = 190  # Set the angle where you want to place the cube
cube_size = 0.2  # Size of the cube (0.2 x 0.2)

# Font sizes for axis labels and ticks
axis_label_fontsize = 50
tick_label_fontsize = 50

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
    
def calculate_average_computation_time(bag_file_path, cmd_vel_topic):
    """Calculate the average computation time between cmd_vel messages for a planner."""
    bag = rosbag.Bag(bag_file_path)
    
    timestamps = []  # List to store timestamps of each cmd_vel message

    # Iterate through the cmd_vel messages and record the timestamps
    for topic, msg, t in bag.read_messages(topics=[cmd_vel_topic]):
        timestamps.append(t.to_sec())  # Store time in seconds

    bag.close()

    # Calculate time differences between consecutive timestamps
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)  # Differences between consecutive timestamps
        average_computation_time = np.mean(time_diffs)  # Average time difference (computation time)
    else:
        average_computation_time = 0  # If less than 2 messages, return 0
    
    return average_computation_time

def process_path_data(path_msg):
    """Process path data from a ROS message."""
    path_x = []
    path_y = []
    for pose in path_msg.poses:
        path_x.append(pose.pose.position.x)
        path_y.append(pose.pose.position.y)
    return path_x, path_y

def calculate_cross_tracking_error(global_path, local_path):
    """Calculate the cross-tracking error between the global path and the local path."""
    errors = []
    for lx, ly in zip(local_path[0], local_path[1]):
        min_distance = min([np.sqrt((lx - gx)**2 + (ly - gy)**2) for gx, gy in zip(global_path[0], global_path[1])])
        errors.append(min_distance)
    return errors

def draw_rotated_rectangle(x, y, width, height, angle, ax):
    """Draw a rotated rectangle."""
    theta = np.radians(angle)
    rect = patches.Rectangle((-width/2, -height/2), width, height, linewidth=1, edgecolor='k', facecolor='k')
    t = plt.gca().transData
    t_rot = Affine2D().rotate(theta).translate(x, y) + t
    rect.set_transform(t_rot)
    ax.add_patch(rect)
    
def calculate_cross_tracking_error(global_path, local_path):
    """Calculate the cross-tracking error between the global path and the local path."""
    errors = []
    for lx, ly in zip(local_path[0], local_path[1]):
        min_distance = min([np.sqrt((lx - gx)**2 + (ly - gy)**2) for gx, gy in zip(global_path[0], global_path[1])])
        errors.append(min_distance)
    return np.mean(errors) if errors else 0  # Return the mean CTE

# Function to calculate the position on the circle based on the angle
def calculate_position_on_circle(center_x, center_y, radius, angle_deg):
    angle_rad = np.radians(angle_deg)
    x = center_x + radius * np.cos(angle_rad)
    y = center_y + radius * np.sin(angle_rad)
    return x, y

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
    
    # Calculate average computation times
    nmpc_avg_computation_time = calculate_average_computation_time(bag_nmpc_path, cmd_vel_topic)
    dwa_avg_computation_time = calculate_average_computation_time(bag_dwa_path, cmd_vel_topic)
    teb_avg_computation_time = calculate_average_computation_time(bag_teb_path, cmd_vel_topic)


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
        (-4.12027, 0.785226, 0.278354, 0),
        (1.5, -0.5, 1.5/2, 0)
    ]

    # Drawing black-filled circles (cylinders)
    for cylinder in cylinders:
        circ = patches.Circle((cylinder[0], cylinder[1]), cylinder[2], linewidth=1, edgecolor='k', facecolor='k')
        ax1.add_patch(circ)
        
    # Drawing rectangles as described in the SDF file
    rectangles = [
        (-1.466790, 0.124105, 0.832545, 0.712184, -0.368363 * 180 / np.pi),
        (0.704279, 1.426650, 0.832545, 0.712184, 0.381073 * 180 / np.pi),
        (1.463960, -0.547771, 0.832545, 0.712184, -0.410182 * 180 / np.pi),
        (3.453640, 1.03602, 0.832545, 0.712184, 0)
    ]

    # Drawing rectangles with rotation
    for rect in rectangles:
        draw_rotated_rectangle(rect[0], rect[1], rect[2], rect[3], rect[4], ax1)

    # Drawing the new door box (as described in cube.urdf)
    draw_rotated_rectangle(5.0, -0.5, 0.01, 0.5, 0, ax1)
    
    # Drawing the dashed circle with diameter = 1 (radius = 0.5)
    dashed_circle = patches.Circle((3, 0), 0.5, linewidth=1, edgecolor='k', facecolor='none', linestyle='--')
    ax1.add_patch(dashed_circle)

    # Draw a black-filled cube along the dashed circle at a specified angle
    cube_x_1, cube_y_1 = calculate_position_on_circle(3, 0, 0.5, cube_angle_on_circle_1)
    draw_rotated_rectangle(cube_x_1, cube_y_1, cube_size, cube_size, 0, ax1)

    # Draw a black-filled cube along the dashed circle at a specified angle
    cube_x_2, cube_y_2 = calculate_position_on_circle(3, 0, 0.5, cube_angle_on_circle_2)
    draw_rotated_rectangle(cube_x_2, cube_y_2, cube_size, cube_size, 0, ax1)

    # Load the global plan from /move_base/NavfnROS/plan
    global_path_x, global_path_y = [], []
    bag_global = rosbag.Bag(bag_nmpc_path)
    for topic, msg, t in bag_global.read_messages(topics=[global_plan_topic]):
        global_path_x, global_path_y = process_path_data(msg)
        break
    bag_global.close()

    # Plot global plan
    ax1.plot(global_path_x, global_path_y, label='Global Plan', color='black', linestyle=':', linewidth=4)

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

    # Plotting the paths with thicker lines and a unique symbol for each path
    ax1.plot(nmpc_path_x, nmpc_path_y, label='nMPC Path', color='green', linestyle='-', linewidth=4, marker='o', markersize=8)
    ax1.plot(dwa_path_x, dwa_path_y, label='DWA Path', color='red', linestyle='--', linewidth=4, marker='s', markersize=8)
    ax1.plot(teb_path_x, teb_path_y, label='TEB Path', color='blue', linestyle='-.', linewidth=4, marker='^', markersize=8)


    # Adjust font sizes using global parameters
    ax1.set_xlabel('X [m]', fontsize=axis_label_fontsize)
    ax1.set_ylabel('Y [m]', fontsize=axis_label_fontsize)
    ax1.set_title('Paths for nMPC, DWA, and TEB with Global Plan', fontsize=axis_label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    ax1.legend(loc='upper right', fontsize=axis_label_fontsize * 2 / 3)  # Adjust legend position and font size

    # Show the plot for the paths
    plt.axis('equal')
    plt.grid(True)
    plt.show()

   # Create figure with three subplots (CTE, Computation Time, Arrival Time)
    # Create figure with three subplots in a single row
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    planners = ['nMPC', 'DWA', 'TEB']
    axis_label_fontsize_2 = axis_label_fontsize / 2
    tick_label_fontsize_2 = tick_label_fontsize / 2

    # ------------------------- CROSS-TRACKING ERROR -------------------------
    # Calculate the cross-tracking error (CTE) for each planner
    nmpc_cte = calculate_cross_tracking_error((global_path_x, global_path_y), (nmpc_path_x, nmpc_path_y))
    dwa_cte = calculate_cross_tracking_error((global_path_x, global_path_y), (dwa_path_x, dwa_path_y))
    teb_cte = calculate_cross_tracking_error((global_path_x, global_path_y), (teb_path_x, teb_path_y))

    cte_values = [nmpc_cte, dwa_cte, teb_cte]

    # Plot CTE in the first subplot
    ax[0].bar(planners, cte_values, color=['green', 'red', 'blue'])
    ax[0].set_xlabel('Planners', fontsize=axis_label_fontsize_2)
    ax[0].set_ylabel('Cross-Tracking Error [m]', fontsize=axis_label_fontsize_2)
    ax[0].set_title('Cross-Tracking Error Comparison', fontsize=axis_label_fontsize_2)
    ax[0].tick_params(axis='both', which='major', labelsize=tick_label_fontsize_2)
    ax[0].grid(True)

    # ------------------------- COMPUTATION TIME -------------------------
    computation_times = [nmpc_avg_computation_time * 1000, dwa_avg_computation_time * 1000, teb_avg_computation_time * 1000]

    # Plot Computation Time in the second subplot
    ax[1].bar(planners, computation_times, color=['green', 'red', 'blue'])
    ax[1].set_xlabel('Planners', fontsize=axis_label_fontsize_2)
    ax[1].set_ylabel('Computation Time [ms]', fontsize=axis_label_fontsize_2)
    ax[1].set_title('Computation Time Comparison', fontsize=axis_label_fontsize_2)
    ax[1].tick_params(axis='both', which='major', labelsize=tick_label_fontsize_2)
    ax[1].grid(True)

    # ------------------------- ARRIVAL TIME -------------------------
    arrival_times = [nmpc_total_time, dwa_total_time, teb_total_time]

    # Plot Arrival Time in the third subplot
    ax[2].bar(planners, arrival_times, color=['green', 'red', 'blue'])
    ax[2].set_xlabel('Planners', fontsize=axis_label_fontsize_2)
    ax[2].set_ylabel('Total Duration [s]', fontsize=axis_label_fontsize_2)
    ax[2].set_title('Arrival Time Comparison', fontsize=axis_label_fontsize_2)
    ax[2].tick_params(axis='both', which='major', labelsize=tick_label_fontsize_2)
    ax[2].grid(True)

    # Adjust layout and show the combined plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
