import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

def interpolate_trajectory(ref_traj, total_prediction_distance, N, robot_max_speed, scale_factor=1.1):
    """
    Interpolates or reduces the given trajectory to exactly N points, including orientation (Yaw).
    Only keeps waypoints that are within the total prediction distance.
    
    :param ref_traj: list of waypoints (x, y, yaw) as reference trajectory
    :param total_prediction_distance: total distance for MPC prediction
    :param N: number of prediction points in MPC
    :param robot_max_speed: maximum speed of the robot
    :param scale_factor: scale factor to adjust reference trajectory distance (default 1.1)
    :return: Interpolated or reduced reference trajectory as a list of waypoints with orientation
    """

    # Berechne die Distanzen zwischen den Punkten der aktuellen Referenz-Trajektorie
    ref_points = np.array(ref_traj)[:, :2]  # Nur x, y für Distanzen berechnen
    yaw_angles = np.array(ref_traj)[:, 2]  # Yaw-Winkel (Orientierung) separat speichern
    distances = np.sqrt(np.sum(np.diff(ref_points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Nur die Waypoints behalten, die innerhalb der totalen Prediction-Distanz liegen
    within_distance_mask = cumulative_distances <= total_prediction_distance
    ref_points_within_distance = ref_points[within_distance_mask]
    yaw_within_distance = yaw_angles[within_distance_mask]
    cumulative_distances_within = cumulative_distances[within_distance_mask]

    # Falls die Anzahl der Punkte nicht ausreicht, füge den letzten Punkt hinzu
    if len(ref_points_within_distance) < 2:
        ref_points_within_distance = np.vstack([ref_points_within_distance, ref_points[-1]])
        yaw_within_distance = np.append(yaw_within_distance, yaw_angles[-1])
        cumulative_distances_within = np.append(cumulative_distances_within, total_prediction_distance)

    # Gesamtdistanz der Referenz-Trajektorie
    total_ref_distance = cumulative_distances_within[-1]

    # Skalierte Distanz zwischen den Punkten in der Referenztrajektorie
    scaled_total_distance = total_ref_distance * scale_factor

    # Berechne die Zeit für jede Vorhersage basierend auf der maximalen Geschwindigkeit
    prediction_distance = total_prediction_distance / N
    T = prediction_distance / robot_max_speed

    # Interpolierte Positionen basierend auf der Gesamtzahl der gewünschten Punkte N
    new_distances = np.linspace(0, total_ref_distance, N)

    # Interpolation für x und y separat
    interp_x = interp1d(cumulative_distances_within, ref_points_within_distance[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(cumulative_distances_within, ref_points_within_distance[:, 1], kind='linear', fill_value="extrapolate")

    new_points_x = interp_x(new_distances)
    new_points_y = interp_y(new_distances)

    # Interpolation der Yaw-Winkel
    interp_yaw = interp1d(cumulative_distances_within, yaw_within_distance, kind='linear', fill_value="extrapolate")
    new_yaw_angles = interp_yaw(new_distances)

    # Erstelle eine neue Trajektorie mit x, y und yaw
    new_trajectory = np.column_stack((new_points_x, new_points_y, new_yaw_angles))

    return new_trajectory, T

# Beispiel für eine Trajektorie mit Wegpunkten (x, y, yaw)
ref_traj = np.array([
    [0, 0, 0],
    [1, 2, 0.5],
    [2, 3, 1.0],
    [4, 5, 1.5],
    [5, 6, 2.0],
    [6, 7, 2.5],
    [7, 7, 2.0],
    [8, 8, 2.0],
    [9, 9, 2.0],
])

# Parameter für die Interpolation
total_prediction_distance = 15  # Gesamtdistanz für die MPC-Prediction
N = 9  # Anzahl der Prediction-Punkte im MPC
current_speed = 0.5  # Aktuelle Geschwindigkeit des Roboters
robot_max_speed = 1.0  # Maximale Geschwindigkeit des Roboters

# Berechnungsstart
start_time = time.time()

# Aufruf der Interpolationsfunktion
new_trajectory, T = interpolate_trajectory(ref_traj, total_prediction_distance, N, current_speed, robot_max_speed)
print(new_trajectory)

# Berechnungsende
end_time = time.time()

# Berechnungsdauer in Sekunden
calculation_time = end_time - start_time

# Plotten der Original- und der interpolierten Trajektorie
plt.figure(figsize=(8, 6))

# Original-Trajektorie (blaue Punkte und Linien) mit größeren Punkten
plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'bo-', markersize=10, label='Original Trajectory')

# Interpolierte Trajektorie (rote Punkte und gestrichelte Linien) mit kleineren Punkten im Vordergrund
plt.plot(new_trajectory[:, 0], new_trajectory[:, 1], 'ro--', markersize=6, zorder=10, label=f'Interpolated Trajectory (N={N})')

# Plotten der Orientierungspfeile für die Originaltrajektorie
for i, (x, y, yaw) in enumerate(ref_traj):
    plt.arrow(x, y, np.cos(yaw), np.sin(yaw), head_width=0.1, color='blue')

# Plotten der Orientierungspfeile für die interpolierte Trajektorie
for i, (x, y, yaw) in enumerate(new_trajectory):
    plt.arrow(x, y, np.cos(yaw), np.sin(yaw), head_width=0.1, color='red', zorder=11)

# Legende
plt.legend()

# Anzeige des Plots
plt.grid(True)
plt.show()

# Ausgabe der Berechnungsdauer in der Konsole
print(f"Die Berechnung der Interpolation dauerte {calculation_time:.6f} Sekunden.")
print(f"Size of new trajectory: {new_trajectory.shape[0]}")
