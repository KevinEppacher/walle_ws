import rosbag
import os
import roslib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tf.transformations import euler_from_quaternion

# Funktion zum Verarbeiten der Pfad-Daten (inkl. Orientierung)
def process_path_data(path_msg):
    path_x = []
    path_y = []
    for pose in path_msg.poses:
        path_x.append(pose.pose.position.x)
        path_y.append(pose.pose.position.y)
    return path_x, path_y

# Funktion zum Zeichnen der Boxen (komplett gefüllt in Schwarz)
def draw_filled_box(ax, x_center, y_center, width, height, yaw=0):
    # Berechne die linke untere Ecke basierend auf dem Mittelpunkt und der Größe
    lower_left_x = x_center - width / 2
    lower_left_y = y_center - height / 2
    # Erstelle das gefüllte Rechteck (Box)
    rect = Rectangle((lower_left_x, lower_left_y), width, height, angle=yaw, linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(rect)

# Hole den Pfad zum ROS-Package amr_control
amr_control_path = roslib.packages.get_pkg_dir('amr_control')

# Vollständige Pfade zu den Rosbag-Dateien
bag_nmpc_path = os.path.join(amr_control_path, 'data/recorded_data_nMPC_2.bag')
bag_dwa_path = os.path.join(amr_control_path, 'data/recorded_data_DWA_3.bag')

# Öffne die Rosbags mit dem vollständigen Pfad
bag_nmpc = rosbag.Bag(bag_nmpc_path)
bag_dwa = rosbag.Bag(bag_dwa_path)

# Extrahiere die Pfad-Daten für nMPC und DWA
nmpc_path_x, nmpc_path_y = [], []
dwa_path_x, dwa_path_y = [], []

for topic, msg, t in bag_nmpc.read_messages(topics=['/robot_path']):
    nmpc_path_x, nmpc_path_y = process_path_data(msg)

for topic, msg, t in bag_dwa.read_messages(topics=['/robot_path_DWA']):
    dwa_path_x, dwa_path_y = process_path_data(msg)

# Schließe die Rosbag-Dateien
bag_nmpc.close()
bag_dwa.close()

# Erstelle die Figur und Achsen
fig, ax = plt.subplots(figsize=(10, 6))

# Plot für nMPC-Pfad
ax.plot(nmpc_path_x, nmpc_path_y, label='nMPC Path', color='green', linestyle='-', linewidth=2)

# Plot für DWA-Pfad
ax.plot(dwa_path_x, dwa_path_y, label='DWA Path', color='red', linestyle='--', linewidth=2)

# Boxen hinzufügen (basierend auf den Screenshots) und komplett in Schwarz füllen
# Box 1
draw_filled_box(ax, 4.928850, -0.5, 0.01, 0.5)

# Box 2
draw_filled_box(ax, 5.184960, -1.439995, 0.15, 2.0)

# Box 3
draw_filled_box(ax, 5.359950, -0.515000, 0.5, 0.15)

# Box 4
draw_filled_box(ax, 5.534950, -1.439995, 0.15, 2.0)

# Box 5
draw_filled_box(ax, 5.174960, 1.364995, 0.155, 0.15)

# Box 6
draw_filled_box(ax, 5.349960, 0.315000, 0.5, 0.15)

# Box 7
draw_filled_box(ax, 5.524950, 1.364995, 2.25, 0.15)

# Gitter hinzufügen
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legenden und Achsenbeschriftung
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Comparison of nMPC and DWA Paths with Black Box Obstacles')

# Achsskalierung für gleichmäßige Darstellung
plt.axis('equal')

# Plot anzeigen
plt.show()
