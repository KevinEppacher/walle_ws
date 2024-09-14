import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
import rosbag
import os
import roslib
from matplotlib.patches import Rectangle

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

# Drawing the environment (walls, cylinders, rectangles)
fig, ax = plt.subplots()

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

# Loading paths from ROSBAGs
amr_control_path = roslib.packages.get_pkg_dir('amr_control')

bag_nmpc_path = os.path.join(amr_control_path, 'data/recorded_data_nMPC_2.bag')
bag_dwa_path = os.path.join(amr_control_path, 'data/recorded_data_DWA_3.bag')

bag_nmpc = rosbag.Bag(bag_nmpc_path)
bag_dwa = rosbag.Bag(bag_dwa_path)

# Extracting the paths
nmpc_path_x, nmpc_path_y = [], []
dwa_path_x, dwa_path_y = [], []

for topic, msg, t in bag_nmpc.read_messages(topics=['/robot_path']):
    nmpc_path_x, nmpc_path_y = process_path_data(msg)

for topic, msg, t in bag_dwa.read_messages(topics=['/robot_path_DWA']):
    dwa_path_x, dwa_path_y = process_path_data(msg)

# Close the ROSBAGs
bag_nmpc.close()
bag_dwa.close()

# Plot the paths
ax.plot(nmpc_path_x, nmpc_path_y, label='nMPC Path', color='green', linestyle='-', linewidth=2)
ax.plot(dwa_path_x, dwa_path_y, label='DWA Path', color='red', linestyle='--', linewidth=2)

# Grid and labels
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('nMPC and DWA Paths with Black Box Obstacles')

# Axis scaling and plot display
plt.axis('equal')
plt.legend()
plt.show()
