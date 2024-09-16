import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
import rosbag
import os
import roslib

# ------------------------- CONFIGURABLE PARAMETERS -------------------------
# ROSBAG file paths
bag_nmpc_filename = 'recorded_data_nMPC_14.bag'
bag_dwa_filename = 'recorded_data_DWA_9.bag'

# ROS topics for paths
topic_nmpc = '/robot_path_DWA'
topic_dwa = '/robot_path_DWA'

# Angle for cube placement along the dashed circle (in degrees)
cube_angle_on_circle = 90  # Set the angle where you want to place the cube
cube_size = 0.2  # Size of the cube (0.2 x 0.2)

# Drawing the environment (walls, cylinders, rectangles)
fig, ax = plt.subplots()

# Font sizes for axis labels and ticks
axis_label_fontsize = 25
tick_label_fontsize = 25

# ---------------------------------------------------------------------------

def draw_rotated_rectangle(x, y, width, height, angle, ax):
    """Draw a rotated rectangle."""
    theta = np.radians(angle)
    # Create a black-filled rectangle centered at (0, 0)
    rect = patches.Rectangle((-width/2, -height/2), width, height, linewidth=1, edgecolor='k', facecolor='k')
    # Apply the rotation and translation
    t = plt.gca().transData
    t_rot = Affine2D().rotate(theta).translate(x, y) + t
    rect.set_transform(t_rot)
    ax.add_patch(rect)

# Process path data from ROSBAG
def process_path_data(path_msg):
    path_x = []
    path_y = []
    for pose in path_msg.poses:
        path_x.append(pose.pose.position.x)
        path_y.append(pose.pose.position.y)
    return path_x, path_y

# Function to calculate the position on the circle based on the angle
def calculate_position_on_circle(center_x, center_y, radius, angle_deg):
    angle_rad = np.radians(angle_deg)
    x = center_x + radius * np.cos(angle_rad)
    y = center_y + radius * np.sin(angle_rad)
    return x, y


# Drawing walls (rectangles) as described in the SDF file
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
    draw_rotated_rectangle(wall[0], wall[1], wall[2], wall[3], wall[4], ax)

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
    ax.add_patch(circ)

# Drawing rectangles as described in the SDF file
rectangles = [
    (-1.466790, 0.124105, 0.832545, 0.712184, -0.368363 * 180 / np.pi),
    (0.704279, 1.426650, 0.832545, 0.712184, 0.381073 * 180 / np.pi),
    (1.463960, -0.547771, 0.832545, 0.712184, -0.410182 * 180 / np.pi),
    (3.453640, 1.03602, 0.832545, 0.712184, 0)
]

# Drawing rectangles with rotation
for rect in rectangles:
    draw_rotated_rectangle(rect[0], rect[1], rect[2], rect[3], rect[4], ax)

# Drawing the new door box (as described in cube.urdf)
draw_rotated_rectangle(5.0, -0.5, 0.01, 0.5, 0, ax)

# Drawing the dashed circle with diameter = 1 (radius = 0.5)
dashed_circle = patches.Circle((3, 0), 0.5, linewidth=1, edgecolor='k', facecolor='none', linestyle='--')
ax.add_patch(dashed_circle)

# Draw a black-filled cube along the dashed circle at a specified angle
cube_x, cube_y = calculate_position_on_circle(3, 0, 0.5, cube_angle_on_circle)  # Position along the circle
draw_rotated_rectangle(cube_x, cube_y, cube_size, cube_size, 0, ax)  # Cube size is 0.2 x 0.2

# Loading paths from ROSBAGs
amr_control_path = roslib.packages.get_pkg_dir('amr_control')

bag_nmpc_path = os.path.join(amr_control_path, f'data/rosbags/{bag_nmpc_filename}')
bag_dwa_path = os.path.join(amr_control_path, f'data/rosbags/{bag_dwa_filename}')

bag_nmpc = rosbag.Bag(bag_nmpc_path)
bag_dwa = rosbag.Bag(bag_dwa_path)

# Extracting the paths
nmpc_path_x, nmpc_path_y = [], []
dwa_path_x, dwa_path_y = [], []

for topic, msg, t in bag_nmpc.read_messages(topics=[topic_nmpc]):
    nmpc_path_x, nmpc_path_y = process_path_data(msg)

for topic, msg, t in bag_dwa.read_messages(topics=[topic_dwa]):
    dwa_path_x, dwa_path_y = process_path_data(msg)

# Close the ROSBAGs
bag_nmpc.close()
bag_dwa.close()

# Plot the paths
ax.plot(nmpc_path_x, nmpc_path_y, label='nMPC Path', color='green', linestyle='-', linewidth=2)
ax.plot(dwa_path_x, dwa_path_y, label='DWA Path', color='red', linestyle='--', linewidth=2)

# Grid and labels
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('X [m]', fontsize=axis_label_fontsize)
plt.ylabel('Y [m]', fontsize=axis_label_fontsize)
plt.title('nMPC and DWA Paths with dynamic Obstacle Avoidance', fontsize=axis_label_fontsize+20)

# Set tick labels font size
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

# Axis scaling and plot display
plt.axis('equal')
# plt.xlim(4.5, 7)
# plt.ylim(-1, 0.5)
plt.legend()
plt.show()
