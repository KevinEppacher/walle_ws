import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np


def draw_rotated_rectangle(x, y, width, height, angle, ax):
    """Draw a rotated rectangle."""
    theta = np.radians(angle)
    # Create a rectangle centered at (0, 0)
    rect = patches.Rectangle((-width/2, -height/2), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # Apply the rotation and translation
    t = plt.gca().transData
    t_rot = Affine2D().rotate(theta).translate(x, y) + t
    rect.set_transform(t_rot)
    ax.add_patch(rect)

# Create figure and axes
fig, ax = plt.subplots()

# Drawing walls (rectangles) as described in the SDF file
# Parameters are: (x, y, width, height, rotation in degrees)
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

# Drawing walls as rectangles with rotation
for wall in walls:
    draw_rotated_rectangle(wall[0], wall[1], wall[2], wall[3], wall[4], ax)

# Drawing cylinders (circles) as described in the SDF file
# Parameters are: (x, y, radius, rotation in degrees)
cylinders = [
    (-2.04473, 1.37495, 0.5, 0),
    (-3.11174, -1.27262, 0.278354, 0),
    (3.55449, -1.48911, 0.278354, 0),
    (-0.526371, -1.58176, 0.278354, 0),
    (-4.12027, 0.785226, 0.278354, 0)
]

# Drawing circles (cylinders) on the plot
for cylinder in cylinders:
    circ = patches.Circle((cylinder[0], cylinder[1]), cylinder[2], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(circ)

# Adjust the limits to ensure all objects are visible
ax.set_xlim(-20, 20)
ax.set_ylim(-5, 5)

# Setting the aspect of the plot to be equal
ax.set_aspect('equal')

# Display the plot
plt.show()
